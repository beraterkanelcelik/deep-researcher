"""Tests for routing functions in chat/nodes.py and chat/research_graph.py."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from chat.nodes import (
    should_continue,
    should_continue_after_save_confirm,
    should_run_research,
)
from chat.research_graph import should_continue_review

pytestmark = pytest.mark.unit


# ── should_continue ──────────────────────────────────────────────────

class TestShouldContinue:
    def test_should_continue_to_tools(self):
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "calculator", "id": "tc-1", "args": {"expression": "2+3"}}],
                )
            ]
        }
        assert should_continue(state) == "tools"

    def test_should_continue_to_deep_research(self):
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "deep_research", "id": "tc-2", "args": {"topic": "AI"}}],
                )
            ]
        }
        assert should_continue(state) == "deep_research"

    def test_should_continue_to_save_confirm(self):
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "save_report", "id": "tc-3", "args": {"report_index": 0}}],
                )
            ]
        }
        assert should_continue(state) == "save_confirm"

    def test_should_continue_to_end(self):
        state = {
            "messages": [AIMessage(content="Hello! How can I help you?")]
        }
        assert should_continue(state) == "end"

    def test_should_continue_mixed_tools_simple_wins(self):
        """When multiple simple tools are called, route to tools."""
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "calculator", "id": "tc-1", "args": {"expression": "1+1"}},
                        {"name": "tavily_search", "id": "tc-2", "args": {"query": "test"}},
                    ],
                )
            ]
        }
        assert should_continue(state) == "tools"

    def test_should_continue_empty_tool_calls(self):
        state = {
            "messages": [AIMessage(content="Done", tool_calls=[])]
        }
        assert should_continue(state) == "end"

    def test_should_continue_deep_research_takes_priority(self):
        """deep_research sentinel takes priority over simple tools."""
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "calculator", "id": "tc-1", "args": {"expression": "1+1"}},
                        {"name": "deep_research", "id": "tc-2", "args": {"topic": "AI"}},
                    ],
                )
            ]
        }
        assert should_continue(state) == "deep_research"


# ── should_run_research ──────────────────────────────────────────────

class TestShouldRunResearch:
    def test_should_run_research_with_topic(self):
        state = {"topic": "AI safety"}
        assert should_run_research(state) == "deep_research"

    def test_should_run_research_without_topic(self):
        state = {"topic": ""}
        assert should_run_research(state) == "agent"

    def test_should_run_research_none_topic(self):
        state = {"topic": None}
        assert should_run_research(state) == "agent"

    def test_should_run_research_missing_key(self):
        state = {}
        assert should_run_research(state) == "agent"


# ── should_continue_review (research subgraph) ──────────────────────

class TestShouldContinueReview:
    def test_should_continue_review_redo(self):
        state = {"status": "redo"}
        assert should_continue_review(state) == "orchestrate"

    def test_should_continue_review_approved(self):
        state = {"status": "approved"}
        assert should_continue_review(state) == "__end__"

    def test_should_continue_review_other_status(self):
        state = {"status": "synthesized"}
        assert should_continue_review(state) == "__end__"


# ── should_continue_after_save_confirm ───────────────────────────────

class TestShouldContinueAfterSaveConfirm:
    def test_should_continue_after_save_confirm_save(self):
        state = {"pending_save": {"title": "Report", "summary": "Test"}}
        assert should_continue_after_save_confirm(state) == "save_to_db"

    def test_should_continue_after_save_confirm_cancel(self):
        state = {"pending_save": None}
        assert should_continue_after_save_confirm(state) == "agent"

    def test_should_continue_after_save_confirm_missing(self):
        state = {}
        assert should_continue_after_save_confirm(state) == "agent"
