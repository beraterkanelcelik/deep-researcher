"""Tests for streaming helper functions in chat/views_stream.py."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from chat.views_stream import (
    GRAPH_NODES,
    format_ai_chunk,
    format_tool_message,
    parse_input_messages,
)

pytestmark = pytest.mark.unit


# ── parse_input_messages ─────────────────────────────────────────────

class TestParseInputMessages:
    def test_parse_human_message(self):
        raw = [{"role": "human", "content": "hello"}]
        result = parse_input_messages(raw)
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "hello"

    def test_parse_human_message_with_type_key(self):
        raw = [{"type": "human", "content": "hi there"}]
        result = parse_input_messages(raw)
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "hi there"

    def test_parse_ai_message(self):
        raw = [{"type": "ai", "content": "I am an AI"}]
        result = parse_input_messages(raw)
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "I am an AI"

    def test_parse_ai_message_with_tool_calls(self):
        raw = [
            {
                "type": "ai",
                "content": "",
                "tool_calls": [
                    {"name": "calculator", "id": "tc-1", "args": {"expression": "2+2"}}
                ],
            }
        ]
        result = parse_input_messages(raw)
        msg = result[0]
        assert isinstance(msg, AIMessage)
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "calculator"

    def test_parse_tool_message(self):
        raw = [
            {
                "type": "tool",
                "content": "4",
                "tool_call_id": "tc-1",
                "name": "calculator",
            }
        ]
        result = parse_input_messages(raw)
        msg = result[0]
        assert isinstance(msg, ToolMessage)
        assert msg.content == "4"
        assert msg.tool_call_id == "tc-1"
        assert msg.name == "calculator"

    def test_parse_system_message(self):
        raw = [{"type": "system", "content": "You are a helpful assistant"}]
        result = parse_input_messages(raw)
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "You are a helpful assistant"

    def test_parse_multiple_messages(self):
        raw = [
            {"type": "human", "content": "What is 2+2?"},
            {"type": "ai", "content": "", "tool_calls": [{"name": "calculator", "id": "tc-1", "args": {"expression": "2+2"}}]},
            {"type": "tool", "content": "4", "tool_call_id": "tc-1", "name": "calculator"},
            {"type": "ai", "content": "The answer is 4."},
        ]
        result = parse_input_messages(raw)
        assert len(result) == 4
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)
        assert isinstance(result[2], ToolMessage)
        assert isinstance(result[3], AIMessage)

    def test_parse_empty_content(self):
        raw = [{"type": "human", "content": ""}]
        result = parse_input_messages(raw)
        assert result[0].content == ""

    def test_parse_defaults_to_human(self):
        raw = [{"content": "no type specified"}]
        result = parse_input_messages(raw)
        assert isinstance(result[0], HumanMessage)


# ── format_ai_chunk ──────────────────────────────────────────────────

class TestFormatAiChunk:
    def test_format_ai_chunk_basic(self):
        result = format_ai_chunk("Hello", "msg-1")
        assert result["type"] == "ai"
        assert result["content"] == "Hello"
        assert result["id"] == "msg-1"
        assert "tool_calls" not in result

    def test_format_ai_chunk_with_tool_calls(self):
        tool_calls = [
            {"id": "tc-1", "name": "calculator", "args": {"expression": "2+2"}}
        ]
        result = format_ai_chunk("Let me calculate", "msg-2", tool_calls=tool_calls)
        assert result["type"] == "ai"
        assert result["content"] == "Let me calculate"
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "calculator"

    def test_format_ai_chunk_empty_content(self):
        result = format_ai_chunk("", "msg-3")
        assert result["content"] == ""

    def test_format_ai_chunk_no_tool_calls_when_none(self):
        result = format_ai_chunk("Test", "msg-4", tool_calls=None)
        assert "tool_calls" not in result


# ── format_tool_message ──────────────────────────────────────────────

class TestFormatToolMessage:
    def test_format_tool_message(self):
        result = format_tool_message("result text", "tc-1", "calculator", "msg-2")
        assert result["type"] == "tool"
        assert result["content"] == "result text"
        assert result["tool_call_id"] == "tc-1"
        assert result["name"] == "calculator"
        assert result["id"] == "msg-2"


# ── GRAPH_NODES constant ────────────────────────────────────────────

class TestGraphNodes:
    def test_graph_nodes_contains_expected(self):
        expected = {
            "retrieve", "agent", "tools", "prepare_research",
            "deep_research", "process_research_result",
            "save_confirm", "save_to_db",
        }
        assert GRAPH_NODES == expected

    def test_subgraph_nodes_not_in_graph_nodes(self):
        subgraph = {"clarify", "orchestrate", "explorer", "synthesize", "review"}
        assert subgraph.isdisjoint(GRAPH_NODES)
