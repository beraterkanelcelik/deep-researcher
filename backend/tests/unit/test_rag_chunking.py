"""Tests for RAG chunking and text extraction in chat/rag/ingest.py."""

import io

import pytest

from chat.rag.ingest import chunk_text, extract_text, extract_text_from_txt

pytestmark = pytest.mark.unit


class TestChunkText:
    def test_chunk_text_basic(self):
        # Create a text that's definitely >500 tokens (roughly 4 chars per token)
        text = "This is a test sentence. " * 200  # ~1000 tokens
        chunks = chunk_text(text)
        assert len(chunks) > 1

    def test_chunk_text_overlap(self):
        """Adjacent chunks should share some content due to overlap."""
        text = "word " * 600  # ~600 tokens
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        if len(chunks) >= 2:
            # The end of chunk 0 and beginning of chunk 1 should overlap
            # Due to token-level overlap, some words should appear in both
            words_0 = set(chunks[0].split()[-60:])
            words_1 = set(chunks[1].split()[:60])
            assert len(words_0 & words_1) > 0

    def test_chunk_text_small_input(self):
        text = "Small text"
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == "Small text"

    def test_chunk_text_exact_boundary(self):
        """Text that's exactly chunk_size tokens should yield one chunk."""
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        # Build a string of exactly 500 tokens
        tokens = encoder.encode("test " * 500)[:500]
        text = encoder.decode(tokens)
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1

    def test_chunk_text_empty_input(self):
        chunks = chunk_text("")
        # Empty text should produce either empty list or single empty chunk
        assert len(chunks) <= 1


class TestExtractTextFromTxt:
    def test_extract_text_from_txt_bytes(self):
        content = b"Hello, this is a test file."
        file = io.BytesIO(content)
        result = extract_text_from_txt(file)
        assert result == "Hello, this is a test file."

    def test_extract_text_from_txt_string(self):
        content = "Hello, this is a string file."
        file = io.StringIO(content)
        result = extract_text_from_txt(file)
        assert result == "Hello, this is a string file."


class TestExtractText:
    def test_extract_text_dispatcher_txt(self):
        content = b"Test content"
        file = io.BytesIO(content)
        result = extract_text(file, "test.txt")
        assert result == "Test content"

    def test_extract_text_dispatcher_md(self):
        content = b"# Markdown content"
        file = io.BytesIO(content)
        result = extract_text(file, "readme.md")
        assert result == "# Markdown content"

    def test_extract_text_dispatcher_py(self):
        content = b"print('hello')"
        file = io.BytesIO(content)
        result = extract_text(file, "script.py")
        assert result == "print('hello')"

    def test_extract_text_unknown_extension(self):
        """Unknown extensions should fall back to text extraction."""
        content = b"some content"
        file = io.BytesIO(content)
        result = extract_text(file, "file.xyz")
        assert result == "some content"
