"""Integration tests for full graph execution with real LLM."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from chat.graph import build_graph

pytestmark = [pytest.mark.integration]


@pytest.fixture
def graph():
    """Compile graph without checkpointer for testing."""
    return build_graph().compile()


class TestSimpleConversation:
    async def test_simple_conversation(self, graph):
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Hello, how are you?")],
                "model": "gpt-4.1-mini",
                "research_reports": [],
                "current_plan": None,
                "pending_save": None,
                "research_request": None,
                "topic": "",
                "depth": "",
                "report": None,
            }
        )
        messages = result["messages"]
        ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
        assert len(ai_msgs) >= 1
        # Should be a plain text response, no tool calls
        last_ai = ai_msgs[-1]
        assert last_ai.content
        assert not last_ai.tool_calls


class TestToolCallCalculator:
    async def test_tool_call_calculator(self, graph):
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="What is 15 * 37?")],
                "model": "gpt-4.1-mini",
                "research_reports": [],
                "current_plan": None,
                "pending_save": None,
                "research_request": None,
                "topic": "",
                "depth": "",
                "report": None,
            }
        )
        messages = result["messages"]
        # Check that calculator was called
        tool_msgs = [m for m in messages if hasattr(m, "name") and m.name == "calculator"]
        assert len(tool_msgs) >= 1
        # Final answer should contain 555
        last_ai = [m for m in messages if isinstance(m, AIMessage)][-1]
        assert "555" in last_ai.content


class TestToolCallTime:
    async def test_tool_call_time(self, graph):
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="What time is it right now?")],
                "model": "gpt-4.1-mini",
                "research_reports": [],
                "current_plan": None,
                "pending_save": None,
                "research_request": None,
                "topic": "",
                "depth": "",
                "report": None,
            }
        )
        messages = result["messages"]
        tool_msgs = [m for m in messages if hasattr(m, "name") and m.name == "get_current_time"]
        assert len(tool_msgs) >= 1


class TestToolCallWebSearch:
    async def test_tool_call_web_search(self, graph):
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Search for the latest Python release version")],
                "model": "gpt-4.1-mini",
                "research_reports": [],
                "current_plan": None,
                "pending_save": None,
                "research_request": None,
                "topic": "",
                "depth": "",
                "report": None,
            }
        )
        messages = result["messages"]
        tool_msgs = [m for m in messages if hasattr(m, "name") and m.name == "tavily_search"]
        assert len(tool_msgs) >= 1


class TestToolCallCreatePlan:
    async def test_tool_call_create_plan(self, graph):
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Create a plan to build a todo app")],
                "model": "gpt-4.1-mini",
                "research_reports": [],
                "current_plan": None,
                "pending_save": None,
                "research_request": None,
                "topic": "",
                "depth": "",
                "report": None,
            }
        )
        messages = result["messages"]
        tool_msgs = [m for m in messages if hasattr(m, "name") and m.name == "create_plan"]
        assert len(tool_msgs) >= 1
        # Verify the plan content has tasks
        import json
        for tm in tool_msgs:
            content = json.loads(tm.content)
            assert "tasks" in content


class TestMultiTurnContext:
    async def test_multi_turn_context(self, graph):
        # First turn
        result1 = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="My name is Alice.")],
                "model": "gpt-4.1-mini",
                "research_reports": [],
                "current_plan": None,
                "pending_save": None,
                "research_request": None,
                "topic": "",
                "depth": "",
                "report": None,
            }
        )
        # Second turn builds on first
        messages = result1["messages"] + [HumanMessage(content="What is my name?")]
        result2 = await graph.ainvoke(
            {
                "messages": messages,
                "model": "gpt-4.1-mini",
                "research_reports": [],
                "current_plan": None,
                "pending_save": None,
                "research_request": None,
                "topic": "",
                "depth": "",
                "report": None,
            }
        )
        last_ai = [m for m in result2["messages"] if isinstance(m, AIMessage)][-1]
        assert "alice" in last_ai.content.lower()


class TestGraphStatePersistence:
    async def test_graph_state_persistence(self, graph):
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Hello")],
                "model": "gpt-4.1-mini",
                "research_reports": [],
                "current_plan": None,
                "pending_save": None,
                "research_request": None,
                "topic": "",
                "depth": "",
                "report": None,
            }
        )
        assert "messages" in result
        assert len(result["messages"]) >= 2
