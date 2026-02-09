"""Shared test fixtures for backend tests."""

import os
import uuid

import pytest
from django.test import AsyncClient

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")


@pytest.fixture
def sample_thread(db):
    """Create a sample Thread in the database."""
    from chat.models import Thread

    return Thread.objects.create(title="Test Thread")


@pytest.fixture
def sample_messages(db, sample_thread):
    """Create sample human + AI messages for a thread."""
    from chat.models import Message

    human = Message.objects.create(
        thread=sample_thread,
        role="human",
        content="Hello, how are you?",
    )
    ai = Message.objects.create(
        thread=sample_thread,
        role="ai",
        content="I'm doing well, thank you! How can I help you today?",
    )
    return [human, ai]


@pytest.fixture
def async_client():
    """Async HTTP client for testing Django views."""
    from httpx import ASGITransport, AsyncClient as HttpxAsyncClient

    from config.asgi import application

    transport = ASGITransport(app=application)
    return HttpxAsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture
def llm():
    """ChatOpenAI instance with real API key for integration tests."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        streaming=True,
    )


@pytest.fixture
def compiled_graph():
    """Build graph without checkpointer for unit tests."""
    from chat.graph import build_graph

    graph = build_graph()
    return graph.compile()


@pytest.fixture
def sample_document(db):
    """Create a Document with Embeddings for RAG tests."""
    from documents.models import Document, Embedding

    doc = Document.objects.create(
        filename="test_doc.txt",
        content="Python is a versatile programming language. It uses snake_case for naming conventions and PascalCase for class names.",
        chunk_index=1,
    )
    # We don't create real embeddings here since that requires API calls;
    # tests that need embeddings should use the @pytest.mark.integration marker
    return doc
