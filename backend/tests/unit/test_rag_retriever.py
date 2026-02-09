"""Tests for RAG retriever in chat/rag/retriever.py."""

import pytest

from chat.rag.retriever import retrieve_documents

pytestmark = [pytest.mark.unit, pytest.mark.django_db]


class TestRetrieveDocuments:
    def test_retrieve_empty_db(self):
        """When no embeddings exist, returns empty list."""
        results = retrieve_documents("any query")
        assert results == []

    @pytest.mark.integration
    def test_retrieve_returns_results(self, sample_document):
        """After ingesting a document, retrieval should return results."""
        from chat.rag.embeddings import generate_embeddings
        from chat.rag.ingest import chunk_text
        from documents.models import Embedding

        chunks = chunk_text(sample_document.content)
        embeddings = generate_embeddings(chunks)

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            Embedding.objects.create(
                document=sample_document,
                content=chunk,
                embedding=embedding,
                metadata={"chunk_index": i, "filename": sample_document.filename},
            )

        results = retrieve_documents("Python naming conventions", top_k=3)
        assert len(results) > 0
        assert "content" in results[0]
        assert "filename" in results[0]
        assert "distance" in results[0]

    @pytest.mark.integration
    def test_retrieve_top_k(self, sample_document):
        """Should respect top_k parameter."""
        from chat.rag.embeddings import generate_embeddings
        from documents.models import Embedding

        # Create multiple chunks
        texts = [
            "Python uses snake_case for variables",
            "Python uses PascalCase for classes",
            "Python supports multiple paradigms",
            "Python has list comprehensions",
            "Python is dynamically typed",
        ]
        embeddings = generate_embeddings(texts)

        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            Embedding.objects.create(
                document=sample_document,
                content=text,
                embedding=embedding,
                metadata={"chunk_index": i, "filename": "test.txt"},
            )

        results = retrieve_documents("naming conventions", top_k=2)
        assert len(results) == 2

    @pytest.mark.integration
    def test_retrieve_relevance_order(self, sample_document):
        """First result should be more relevant (lower distance) than last."""
        from chat.rag.embeddings import generate_embeddings
        from documents.models import Embedding

        texts = [
            "Machine learning algorithms for image classification",
            "Python naming conventions use snake_case",
            "Database indexing strategies for PostgreSQL",
        ]
        embeddings = generate_embeddings(texts)

        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            Embedding.objects.create(
                document=sample_document,
                content=text,
                embedding=embedding,
                metadata={"chunk_index": i, "filename": "test.txt"},
            )

        results = retrieve_documents("Python naming style", top_k=3)
        assert len(results) == 3
        assert results[0]["distance"] <= results[-1]["distance"]
