"""Tests for DRF serializers in chat/serializers.py."""

import pytest

from chat.models import Message, Thread
from chat.serializers import MessageSerializer, ThreadDetailSerializer, ThreadSerializer

pytestmark = [pytest.mark.unit, pytest.mark.django_db]


class TestThreadSerializer:
    def test_thread_serializer_output(self, sample_thread):
        serializer = ThreadSerializer(sample_thread)
        data = serializer.data
        assert "thread_id" in data
        assert str(sample_thread.id) == str(data["thread_id"])
        assert data["title"] == "Test Thread"
        assert "created_at" in data
        assert "updated_at" in data

    def test_thread_serializer_fields(self):
        fields = ThreadSerializer.Meta.fields
        assert "thread_id" in fields
        assert "title" in fields
        assert "metadata" in fields


class TestMessageSerializer:
    def test_message_serializer_output(self, sample_messages):
        msg = sample_messages[0]
        serializer = MessageSerializer(msg)
        data = serializer.data
        assert "type" in data
        assert data["type"] == "human"
        assert data["content"] == "Hello, how are you?"
        assert "created_at" in data

    def test_message_serializer_type_alias(self, sample_messages):
        """The 'type' field should be an alias for 'role'."""
        human_msg = sample_messages[0]
        ai_msg = sample_messages[1]
        assert MessageSerializer(human_msg).data["type"] == "human"
        assert MessageSerializer(ai_msg).data["type"] == "ai"


class TestThreadDetailSerializer:
    def test_thread_detail_includes_messages(self, sample_thread, sample_messages):
        serializer = ThreadDetailSerializer(sample_thread)
        data = serializer.data
        assert "messages" in data
        assert len(data["messages"]) == 2
        assert data["messages"][0]["type"] == "human"
        assert data["messages"][1]["type"] == "ai"
