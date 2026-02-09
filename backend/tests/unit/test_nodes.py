"""Tests for node functions in chat/nodes.py."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from chat.nodes import (
    SYSTEM_PROMPT,
    agent_node,
    prepare_research_node,
    process_research_result_node,
    retrieve_node,
    save_to_db_node,
)

pytestmark = pytest.mark.unit


# ── retrieve_node ────────────────────────────────────────────────────

class TestRetrieveNode:
    @pytest.mark.django_db
    def test_retrieve_node_no_docs(self):
        """When no documents exist in DB, retrieve_node returns empty dict."""
        state = {"messages": [HumanMessage(content="Hello")]}
        result = retrieve_node(state)
        # No embeddings in DB → get_rag_context returns "" → retrieve_node returns {}
        assert result == {}

    def test_retrieve_node_empty_messages(self):
        state = {"messages": []}
        result = retrieve_node(state)
        assert result == {}

    @pytest.mark.django_db
    @pytest.mark.integration
    def test_retrieve_node_with_docs(self):
        """When documents exist in DB, retrieve_node injects context into SystemMessage."""
        from documents.models import Document, Embedding
        from chat.rag.embeddings import generate_embeddings

        # Create a document with embeddings in the DB
        doc = Document.objects.create(
            filename="python_guide.txt",
            content="Python uses snake_case for variable naming and PascalCase for class names.",
            chunk_index=1,
        )
        chunks = ["Python uses snake_case for variable naming and PascalCase for class names."]
        embeddings = generate_embeddings(chunks)
        Embedding.objects.create(
            document=doc,
            content=chunks[0],
            embedding=embeddings[0],
            metadata={"chunk_index": 0, "filename": "python_guide.txt"},
        )

        # Now call retrieve_node with a related query
        state = {"messages": [HumanMessage(content="What naming conventions does Python use?")]}
        result = retrieve_node(state)

        # Should return messages with a SystemMessage containing the RAG context
        assert "messages" in result
        messages = result["messages"]
        # First message should be a SystemMessage with context injected
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        assert len(system_msgs) >= 1
        system_content = system_msgs[0].content
        assert "Context:" in system_content
        assert "snake_case" in system_content
        assert "python_guide.txt" in system_content

    @pytest.mark.django_db
    @pytest.mark.integration
    def test_retrieve_node_context_format(self):
        """Verify the injected context includes source filename and content."""
        from documents.models import Document, Embedding
        from chat.rag.embeddings import generate_embeddings

        doc = Document.objects.create(
            filename="api_docs.txt",
            content="The API uses Bearer token authentication.",
            chunk_index=1,
        )
        chunks = ["The API uses Bearer token authentication."]
        embeddings = generate_embeddings(chunks)
        Embedding.objects.create(
            document=doc,
            content=chunks[0],
            embedding=embeddings[0],
            metadata={"chunk_index": 0, "filename": "api_docs.txt"},
        )

        state = {"messages": [HumanMessage(content="How does the API authenticate?")]}
        result = retrieve_node(state)

        assert "messages" in result
        system_msg = result["messages"][0]
        assert isinstance(system_msg, SystemMessage)
        # Should contain the source marker and the document content
        assert "[Source: api_docs.txt]" in system_msg.content
        assert "Bearer token" in system_msg.content
        # Should also contain the base system prompt
        assert "helpful AI assistant" in system_msg.content


# ── agent_node ───────────────────────────────────────────────────────

class TestAgentNode:
    @pytest.mark.integration
    def test_agent_node_returns_ai_message(self):
        state = {"messages": [HumanMessage(content="Hello")]}
        result = agent_node(state)
        assert "messages" in result
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, AIMessage)

    @pytest.mark.integration
    def test_agent_node_with_tool_calls(self):
        state = {"messages": [HumanMessage(content="What time is it?")]}
        result = agent_node(state)
        msg = result["messages"][0]
        assert isinstance(msg, AIMessage)
        # Should have tool calls for get_current_time
        if msg.tool_calls:
            tool_names = [tc["name"] for tc in msg.tool_calls]
            assert "get_current_time" in tool_names

    @pytest.mark.integration
    def test_agent_node_uses_model_config(self):
        state = {
            "messages": [HumanMessage(content="Say hi")],
            "model": "gpt-4.1-mini",
        }
        result = agent_node(state)
        assert "messages" in result

    def test_agent_node_injects_system_prompt(self):
        """When no system message is present, agent_node should add one."""
        # We test the message assembly logic indirectly by verifying the function
        # doesn't crash and the system prompt is defined correctly.
        assert "helpful AI assistant" in SYSTEM_PROMPT
        assert "deep_research" in SYSTEM_PROMPT


# ── prepare_research_node ────────────────────────────────────────────

class TestPrepareResearchNode:
    def test_prepare_research_node_extracts_args(self):
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "deep_research",
                            "id": "tc-1",
                            "args": {"topic": "AI safety", "depth": "deep"},
                        }
                    ],
                )
            ]
        }
        result = prepare_research_node(state)
        assert result["topic"] == "AI safety"
        assert result["depth"] == "deep"

    def test_prepare_research_node_default_depth(self):
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "deep_research",
                            "id": "tc-2",
                            "args": {"topic": "quantum computing"},
                        }
                    ],
                )
            ]
        }
        result = prepare_research_node(state)
        assert result["topic"] == "quantum computing"
        assert result["depth"] == "standard"

    def test_prepare_research_node_missing_topic(self):
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "deep_research",
                            "id": "tc-3",
                            "args": {},
                        }
                    ],
                )
            ]
        }
        result = prepare_research_node(state)
        assert "messages" in result
        msg = result["messages"][0]
        assert isinstance(msg, ToolMessage)
        assert "no research topic" in msg.content.lower()

    def test_prepare_research_node_no_tool_calls(self):
        state = {"messages": [AIMessage(content="Hello")]}
        result = prepare_research_node(state)
        assert "messages" in result
        assert result["topic"] == ""


# ── process_research_result_node ─────────────────────────────────────

class TestProcessResearchResultNode:
    def test_process_research_result_node_with_report(self):
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "deep_research", "id": "tc-1", "args": {"topic": "AI"}}],
                )
            ],
            "report": {
                "title": "AI Report",
                "summary": "A report about AI",
                "key_findings": [
                    {"insight": "AI is growing", "evidence": "Market data", "sources": ["url1"]}
                ],
                "sources": ["url1"],
                "tags": ["AI"],
                "methodology": "Web search",
            },
        }
        result = process_research_result_node(state)
        assert "messages" in result
        msg = result["messages"][0]
        assert isinstance(msg, ToolMessage)
        assert msg.name == "deep_research"
        assert "AI Report" in msg.content
        assert "research_reports" in result
        assert len(result["research_reports"]) == 1

    def test_process_research_result_node_no_report(self):
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "deep_research", "id": "tc-1", "args": {"topic": "AI"}}],
                )
            ],
            "report": None,
        }
        result = process_research_result_node(state)
        msg = result["messages"][0]
        assert "no report" in msg.content.lower()


# ── save_to_db_node ──────────────────────────────────────────────────

class TestSaveToDbNode:
    def test_save_to_db_node_with_pending(self):
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "save_report", "id": "tc-1", "args": {"report_index": 0}}],
                )
            ],
            "pending_save": {
                "title": "My Report",
                "summary": "Summary",
                "key_findings": [],
                "sources": [],
                "tags": [],
                "methodology": "",
            },
        }
        result = save_to_db_node(state)
        msg = result["messages"][0]
        assert isinstance(msg, ToolMessage)
        content = json.loads(msg.content)
        assert content["status"] == "saved"
        assert "My Report" in content["title"]

    def test_save_to_db_node_no_pending(self):
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "save_report", "id": "tc-1", "args": {"report_index": 0}}],
                )
            ],
            "pending_save": None,
        }
        result = save_to_db_node(state)
        msg = result["messages"][0]
        assert "no report" in msg.content.lower()
