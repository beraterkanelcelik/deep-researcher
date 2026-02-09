"""Integration tests for the full RAG pipeline."""

import io

import pytest

from documents.models import Document, Embedding

pytestmark = [pytest.mark.integration, pytest.mark.django_db]


class TestUploadAndRetrieve:
    def test_upload_and_retrieve(self):
        """Upload a text file, then verify retrieval returns matching content."""
        from chat.rag.ingest import ingest_document
        from chat.rag.retriever import retrieve_documents

        content = (
            "Django REST Framework provides powerful serialization and "
            "authentication mechanisms for building web APIs. "
            "It supports token-based and session-based authentication."
        )
        file = io.BytesIO(content.encode("utf-8"))
        doc, chunk_count = ingest_document(file, "api_guide.txt")

        assert chunk_count >= 1

        results = retrieve_documents("authentication in Django REST", top_k=3)
        assert len(results) >= 1
        # The retrieved content should relate to our uploaded document
        found = any("authentication" in r["content"].lower() for r in results)
        assert found

    def test_upload_creates_records(self):
        """Verify Document and Embedding records are created."""
        from chat.rag.ingest import ingest_document

        content = "Test document for record creation. " * 50
        file = io.BytesIO(content.encode("utf-8"))
        doc, chunk_count = ingest_document(file, "records_test.txt")

        assert Document.objects.filter(id=doc.id).exists()
        assert Embedding.objects.filter(document=doc).count() == chunk_count


class TestRagImprovesAnswer:
    async def test_rag_improves_answer(self):
        """Upload a domain-specific doc, then verify the agent uses it."""
        from chat.graph import build_graph
        from chat.rag.ingest import ingest_document
        from langchain_core.messages import AIMessage, HumanMessage

        # Upload a document with specific domain knowledge
        content = (
            "The company policy states that all employees must use "
            "two-factor authentication (2FA) with hardware security keys. "
            "The approved key models are YubiKey 5 and Google Titan."
        )
        file = io.BytesIO(content.encode("utf-8"))
        ingest_document(file, "security_policy.txt")

        # Ask a question about the document
        graph = build_graph().compile()
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="What are the approved security key models according to company policy?")],
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

        last_ai = [m for m in result["messages"] if isinstance(m, AIMessage)][-1]
        content_lower = last_ai.content.lower()
        # Should reference the specific keys from the document
        assert "yubikey" in content_lower or "titan" in content_lower
