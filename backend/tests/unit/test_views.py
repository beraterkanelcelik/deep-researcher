"""Tests for REST endpoints in chat/views.py."""

import uuid

import pytest
from django.test import Client

from chat.models import Message, Thread

pytestmark = [pytest.mark.unit, pytest.mark.django_db]


class TestThreadListView:
    def test_create_thread(self, client: Client):
        response = client.post("/api/threads/", content_type="application/json")
        assert response.status_code == 201
        data = response.json()
        assert "thread_id" in data

    def test_list_threads(self, client: Client, sample_thread):
        response = client.get("/api/threads/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1


class TestThreadDetailView:
    def test_get_thread(self, client: Client, sample_thread):
        response = client.get(f"/api/threads/{sample_thread.id}/")
        assert response.status_code == 200
        data = response.json()
        assert str(data["thread_id"]) == str(sample_thread.id)

    def test_get_thread_not_found(self, client: Client):
        bad_id = uuid.uuid4()
        response = client.get(f"/api/threads/{bad_id}/")
        assert response.status_code == 404

    def test_delete_thread(self, client: Client, sample_thread):
        response = client.delete(f"/api/threads/{sample_thread.id}/")
        assert response.status_code == 204
        assert not Thread.objects.filter(id=sample_thread.id).exists()


class TestThreadStateView:
    def test_thread_state(self, client: Client, sample_thread, sample_messages):
        response = client.get(f"/api/threads/{sample_thread.id}/state")
        assert response.status_code == 200
        data = response.json()
        assert "values" in data
        assert "messages" in data["values"]
        assert "tasks" in data
        messages = data["values"]["messages"]
        assert len(messages) == 2
        assert messages[0]["type"] == "human"
        assert messages[1]["type"] == "ai"

    def test_thread_state_not_found(self, client: Client):
        bad_id = uuid.uuid4()
        response = client.get(f"/api/threads/{bad_id}/state")
        assert response.status_code == 404
