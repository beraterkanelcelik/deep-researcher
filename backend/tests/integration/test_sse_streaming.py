"""Integration tests for SSE streaming endpoints."""

import json
import uuid

import pytest

from chat.models import Thread

pytestmark = [pytest.mark.integration, pytest.mark.django_db]


def parse_sse_events(content: str) -> list[dict]:
    """Parse SSE text into a list of {event, data} dicts."""
    events = []
    current_event = None
    current_data = []

    for line in content.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:]
        elif line.startswith("data: "):
            current_data.append(line[6:])
        elif line == "" and current_event is not None:
            data_str = "\n".join(current_data)
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = data_str
            events.append({"event": current_event, "data": data})
            current_event = None
            current_data = []

    return events


class TestStreamEndpoint:
    async def test_stream_returns_sse_events(self, async_client):
        thread = await Thread.objects.acreate()
        response = await async_client.post(
            f"/api/threads/{thread.id}/runs/stream",
            json={
                "input": {"messages": [{"role": "human", "content": "Say hello"}]},
                "model": "gpt-4.1-mini",
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        content = response.text
        events = parse_sse_events(content)
        assert len(events) > 0
        event_types = [e["event"] for e in events]
        assert "metadata" in event_types

    async def test_stream_metadata_event(self, async_client):
        thread = await Thread.objects.acreate()
        response = await async_client.post(
            f"/api/threads/{thread.id}/runs/stream",
            json={
                "input": {"messages": [{"role": "human", "content": "Hi"}]},
                "model": "gpt-4.1-mini",
            },
        )
        events = parse_sse_events(response.text)
        metadata_events = [e for e in events if e["event"] == "metadata"]
        assert len(metadata_events) >= 1
        assert "run_id" in metadata_events[0]["data"]

    async def test_stream_ends_with_end_event(self, async_client):
        thread = await Thread.objects.acreate()
        response = await async_client.post(
            f"/api/threads/{thread.id}/runs/stream",
            json={
                "input": {"messages": [{"role": "human", "content": "Hello"}]},
                "model": "gpt-4.1-mini",
            },
        )
        events = parse_sse_events(response.text)
        assert events[-1]["event"] == "end"

    async def test_stream_node_status_events(self, async_client):
        thread = await Thread.objects.acreate()
        response = await async_client.post(
            f"/api/threads/{thread.id}/runs/stream",
            json={
                "input": {"messages": [{"role": "human", "content": "Hello"}]},
                "model": "gpt-4.1-mini",
            },
        )
        events = parse_sse_events(response.text)
        node_events = [e for e in events if e["event"] == "node/status"]
        assert len(node_events) >= 1
        # Should have at least __graph__ started
        graph_events = [e for e in node_events if e["data"].get("node") == "__graph__"]
        assert len(graph_events) >= 1

    async def test_stream_messages_partial_then_complete(self, async_client):
        thread = await Thread.objects.acreate()
        response = await async_client.post(
            f"/api/threads/{thread.id}/runs/stream",
            json={
                "input": {"messages": [{"role": "human", "content": "Hello, how are you?"}]},
                "model": "gpt-4.1-mini",
            },
        )
        events = parse_sse_events(response.text)
        event_types = [e["event"] for e in events]
        # Should have partial chunks before complete
        has_partial = "messages/partial" in event_types
        has_complete = "messages/complete" in event_types
        assert has_complete
        if has_partial:
            first_partial = event_types.index("messages/partial")
            last_complete = len(event_types) - 1 - event_types[::-1].index("messages/complete")
            assert first_partial < last_complete
