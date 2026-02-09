"""Tests for RAG embeddings in chat/rag/embeddings.py."""

import pytest

from chat.rag.embeddings import generate_embedding, generate_embeddings

pytestmark = [pytest.mark.unit, pytest.mark.integration]


class TestGenerateEmbedding:
    def test_generate_embedding_dimension(self):
        result = generate_embedding("test text")
        assert isinstance(result, list)
        assert len(result) == 1536
        assert all(isinstance(x, float) for x in result)

    def test_generate_embedding_deterministic(self):
        result1 = generate_embedding("exact same text")
        result2 = generate_embedding("exact same text")
        # Embeddings for same text should be very close (API has minor float variance)
        diff = sum(abs(a - b) for a, b in zip(result1, result2))
        assert diff < 0.1

    def test_similar_texts_close_vectors(self):
        """Similar texts should produce similar embeddings."""
        e1 = generate_embedding("Python is a programming language")
        e2 = generate_embedding("Python is a coding language")
        # Cosine similarity
        dot = sum(a * b for a, b in zip(e1, e2))
        norm1 = sum(a * a for a in e1) ** 0.5
        norm2 = sum(b * b for b in e2) ** 0.5
        similarity = dot / (norm1 * norm2)
        assert similarity > 0.8


class TestGenerateEmbeddings:
    def test_generate_embeddings_batch(self):
        texts = ["apple", "banana", "cherry"]
        results = generate_embeddings(texts)
        assert len(results) == 3
        assert all(len(v) == 1536 for v in results)
        assert all(isinstance(v, list) for v in results)
