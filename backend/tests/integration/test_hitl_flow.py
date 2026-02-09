"""Integration tests for HITL interrupt and resume cycles."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from chat.graph import build_graph
from chat.nodes import save_confirm_node

pytestmark = [pytest.mark.integration]


class TestSaveConfirmInterrupt:
    def test_save_confirm_with_reports(self):
        """save_confirm_node should call interrupt() when reports exist."""
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "save_report", "id": "tc-1", "args": {"report_index": 0}}],
                )
            ],
            "research_reports": [
                {
                    "title": "Test Report",
                    "summary": "A test report summary",
                    "key_findings": [],
                    "sources": [],
                    "tags": [],
                    "methodology": "Testing",
                }
            ],
        }
        # save_confirm_node calls interrupt() which raises GraphInterrupt
        # when run outside of a proper graph context
        from langgraph.errors import GraphInterrupt

        with pytest.raises(GraphInterrupt):
            save_confirm_node(state)

    def test_save_confirm_no_reports(self):
        """When no reports, save_confirm_node returns error message."""
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "save_report", "id": "tc-1", "args": {"report_index": 0}}],
                )
            ],
            "research_reports": [],
        }
        result = save_confirm_node(state)
        assert "messages" in result
        msg = result["messages"][0]
        assert isinstance(msg, ToolMessage)
        assert "no research reports" in msg.content.lower()
        assert result["pending_save"] is None


class TestResearchSubgraphInterrupt:
    def test_clarify_node_interrupt(self):
        """clarify_node should trigger interrupt for topic selection."""
        from langgraph.errors import GraphInterrupt

        from chat.research_graph import clarify_node

        state = {"topic": "Artificial Intelligence", "depth": "standard"}
        with pytest.raises(GraphInterrupt):
            clarify_node(state)


class TestGraphBuild:
    def test_graph_builds_successfully(self):
        """build_graph() should return a valid StateGraph."""
        graph = build_graph()
        compiled = graph.compile()
        assert compiled is not None

    def test_graph_has_all_nodes(self):
        """Compiled graph should contain all expected nodes."""
        graph = build_graph()
        compiled = graph.compile()
        # Get node names from the graph
        node_names = set(compiled.get_graph().nodes.keys())
        expected = {
            "retrieve", "agent", "tools", "prepare_research",
            "deep_research", "process_research_result",
            "save_confirm", "save_to_db",
        }
        # Graph also includes __start__ and __end__
        assert expected.issubset(node_names)
