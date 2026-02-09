import asyncio
import os

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .graph import get_graph_with_checkpointer
from .models import Thread, Message
from .serializers import ThreadSerializer, MessageSerializer


@api_view(["GET", "POST"])
def thread_list(request):
    if request.method == "GET":
        threads = Thread.objects.all()
        serializer = ThreadSerializer(threads, many=True)
        return Response(serializer.data)

    if request.method == "POST":
        thread = Thread.objects.create()
        serializer = ThreadSerializer(thread)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


@api_view(["GET", "DELETE"])
def thread_detail(request, thread_id):
    try:
        thread = Thread.objects.get(id=thread_id)
    except Thread.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == "GET":
        serializer = ThreadSerializer(thread)
        return Response(serializer.data)

    if request.method == "DELETE":
        thread.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(["GET"])
def thread_state(request, thread_id):
    """Return thread state in LangGraph Cloud format, including interrupt info."""
    try:
        thread = Thread.objects.get(id=thread_id)
    except Thread.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    messages = thread.messages.all()
    message_list = []
    for msg in messages:
        message_data = {
            "type": msg.role,
            "content": msg.content,
            "id": str(msg.id),
        }
        if msg.tool_calls:
            message_data["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            message_data["tool_call_id"] = msg.tool_call_id
        if msg.name:
            message_data["name"] = msg.name
        message_list.append(message_data)

    # Check for active interrupts from the graph checkpointer
    tasks = []
    try:
        async def _get_interrupt_info():
            async with get_graph_with_checkpointer() as compiled_graph:
                config = {"configurable": {"thread_id": str(thread_id)}}
                graph_state = await compiled_graph.aget_state(config)
                if graph_state and hasattr(graph_state, "tasks") and graph_state.tasks:
                    for task in graph_state.tasks:
                        task_info = {}
                        if hasattr(task, "interrupts") and task.interrupts:
                            task_info["interrupts"] = [
                                intr.value if hasattr(intr, "value") else intr
                                for intr in task.interrupts
                            ]
                        if task_info:
                            tasks.append(task_info)
            return tasks

        loop = asyncio.new_event_loop()
        try:
            interrupt_tasks = loop.run_until_complete(_get_interrupt_info())
        finally:
            loop.close()
    except Exception:
        interrupt_tasks = []

    return Response(
        {
            "values": {"messages": message_list},
            "tasks": interrupt_tasks,
        }
    )
