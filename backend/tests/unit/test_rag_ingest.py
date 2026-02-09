"""Tests for RAG ingest pipeline in chat/rag/ingest.py."""

import io

import pytest

from chat.rag.ingest import ingest_document

pytestmark = [pytest.mark.integration, pytest.mark.django_db]


class TestIngestDocument:
    def test_ingest_text_file(self):
        """Full ingest pipeline for a text file."""
        from documents.models import Document, Embedding

        content = "Python best practices guide. " * 100
        file = io.BytesIO(content.encode("utf-8"))

        doc, chunk_count = ingest_document(file, "best_practices.txt")

        assert isinstance(doc, Document)
        assert doc.filename == "best_practices.txt"
        assert chunk_count > 0
        assert Embedding.objects.filter(document=doc).count() == chunk_count

    def test_ingest_small_file(self):
        """A small file should produce a single chunk."""
        content = "Short content."
        file = io.BytesIO(content.encode("utf-8"))

        doc, chunk_count = ingest_document(file, "short.txt")

        assert chunk_count == 1
        assert doc.content == "Short content."
