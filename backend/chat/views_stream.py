import json
import uuid

from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from asgiref.sync import sync_to_async
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Command

from .graph import get_graph_with_checkpointer
from .models import Message, ResearchReport as ResearchReportModel, Thread

# Main graph node names — anything not in this set is treated as a subgraph node
GRAPH_NODES = {"retrieve", "agent", "tools", "prepare_research", "deep_research", "process_research_result", "save_confirm", "save_to_db"}


def parse_input_messages(raw_messages):
    """Convert raw message dicts from the frontend to LangChain message objects."""
    messages = []
    for msg in raw_messages:
        msg_type = msg.get("type", msg.get("role", "human"))
        content = msg.get("content", "")

        if msg_type == "human":
            messages.append(HumanMessage(content=content))
        elif msg_type == "ai":
            messages.append(
                AIMessage(
                    content=content,
                    tool_calls=msg.get("tool_calls", []),
                )
            )
        elif msg_type == "tool":
            messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=msg.get("tool_call_id", ""),
                    name=msg.get("name", ""),
                )
            )
        elif msg_type == "system":
            messages.append(SystemMessage(content=content))

    return messages


def format_ai_chunk(chunk_content, msg_id, tool_calls=None):
    """Format an AI message chunk in LangGraph Cloud format."""
    msg = {
        "type": "ai",
        "content": chunk_content,
        "id": msg_id,
    }
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
            }
            for tc in tool_calls
        ]
    return msg


def format_tool_message(content, tool_call_id, name, msg_id):
    """Format a tool message in LangGraph Cloud format."""
    return {
        "type": "tool",
        "content": content,
        "tool_call_id": tool_call_id,
        "name": name,
        "id": msg_id,
    }


async def _stream_graph(compiled_graph, graph_input, config, thread_id, run_id):
    """Shared async generator: run graph and yield SSE events.

    Uses astream() with stream_mode=["messages", "updates"] and subgraphs=True
    to support interrupt detection and subgraph node tracking.
    """
    yield f"event: metadata\ndata: {json.dumps({'run_id': run_id})}\n\n"
    yield f"event: node/status\ndata: {json.dumps({'node': '__graph__', 'status': 'started'})}\n\n"

    accumulated_content = ""
    current_msg_id = f"msg-{uuid.uuid4().hex[:8]}"
    pending_save_report = None
    active_nodes = set()
    # Track pending tool calls from AI messages: {tool_name: [tool_call_id, ...]}
    pending_tool_calls = {}

    try:
        async for event in compiled_graph.astream_events(
            graph_input,
            config=config,
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                # Only stream tokens from the agent node — skip inner LLM calls
                # from tools (e.g. create_plan) and subgraph nodes.
                node = event.get("metadata", {}).get("langgraph_node")
                if node != "agent":
                    continue
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    accumulated_content += chunk.content
                    msg_data = format_ai_chunk(accumulated_content, current_msg_id)
                    yield f"event: messages/partial\ndata: {json.dumps([msg_data])}\n\n"

            elif kind == "on_chat_model_end":
                # Only process completions from the agent node.
                node = event.get("metadata", {}).get("langgraph_node")
                if node != "agent":
                    continue
                output = event["data"]["output"]

                if hasattr(output, "tool_calls") and output.tool_calls:
                    tool_calls_formatted = [
                        {
                            "id": tc["id"],
                            "name": tc["name"],
                            "args": tc["args"],
                        }
                        for tc in output.tool_calls
                    ]
                    # Track tool call IDs so on_tool_end can match them
                    for tc in output.tool_calls:
                        pending_tool_calls.setdefault(tc["name"], []).append(tc["id"])

                    msg_data = format_ai_chunk(
                        accumulated_content or "",
                        current_msg_id,
                        tool_calls=tool_calls_formatted,
                    )
                    yield f"event: messages/complete\ndata: {json.dumps([msg_data])}\n\n"

                    await sync_to_async(Message.objects.create)(
                        thread_id=thread_id,
                        role="ai",
                        content=accumulated_content or "",
                        tool_calls=tool_calls_formatted,
                    )
                else:
                    msg_data = format_ai_chunk(accumulated_content, current_msg_id)
                    yield f"event: messages/complete\ndata: {json.dumps([msg_data])}\n\n"

                    if accumulated_content:
                        await sync_to_async(Message.objects.create)(
                            thread_id=thread_id,
                            role="ai",
                            content=accumulated_content,
                        )

                accumulated_content = ""
                current_msg_id = f"msg-{uuid.uuid4().hex[:8]}"

            elif kind == "on_tool_end":
                tool_name = event.get("name", "")

                # Only process tool events that match a pending AI tool call.
                # This filters out inner/nested tool calls (e.g. TavilySearchResults
                # inside our tavily_search function) that leak through astream_events.
                if tool_name not in pending_tool_calls or not pending_tool_calls[tool_name]:
                    continue

                tool_call_id = pending_tool_calls[tool_name].pop(0)
                tool_output = event["data"].get("output", "")
                tool_msg_id = f"msg-{uuid.uuid4().hex[:8]}"

                content = str(tool_output)
                if hasattr(tool_output, "content"):
                    content = tool_output.content

                tool_msg = format_tool_message(
                    content=content,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    msg_id=tool_msg_id,
                )
                yield f"event: messages/complete\ndata: {json.dumps([tool_msg])}\n\n"

                await sync_to_async(Message.objects.create)(
                    thread_id=thread_id,
                    role="tool",
                    content=content,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                )

            elif kind == "on_chain_start":
                node_name = event.get("metadata", {}).get("langgraph_node")
                if node_name and event.get("name") == node_name:
                    is_subgraph = node_name not in GRAPH_NODES
                    data = {"node": node_name, "status": "active"}
                    if is_subgraph:
                        data["subgraph"] = True
                    yield f"event: node/status\ndata: {json.dumps(data)}\n\n"
                    active_nodes.add(node_name)

            elif kind == "on_chain_end":
                node_name = event.get("metadata", {}).get("langgraph_node")
                if node_name and event.get("name") == node_name:
                    is_subgraph = node_name not in GRAPH_NODES
                    data = {"node": node_name, "status": "completed"}
                    if is_subgraph:
                        data["subgraph"] = True
                    yield f"event: node/status\ndata: {json.dumps(data)}\n\n"
                    active_nodes.discard(node_name)

        # Check for interrupts after stream completes
        graph_state = await compiled_graph.aget_state(config)
        if graph_state and hasattr(graph_state, "tasks") and graph_state.tasks:
            for task in graph_state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    for intr in task.interrupts:
                        interrupt_value = intr.value if hasattr(intr, "value") else intr
                        yield f"event: interrupt\ndata: {json.dumps(interrupt_value if isinstance(interrupt_value, dict) else {'value': str(interrupt_value)})}\n\n"

        # Persist pending_save report if present
        if hasattr(graph_state, "values"):
            state_values = graph_state.values
            pending = state_values.get("pending_save")
            if pending and isinstance(pending, dict):
                await sync_to_async(ResearchReportModel.objects.create)(
                    thread_id=thread_id,
                    title=pending.get("title", "Untitled"),
                    summary=pending.get("summary", ""),
                    key_findings=pending.get("key_findings", []),
                    sources=pending.get("sources", []),
                    tags=pending.get("tags", []),
                    methodology=pending.get("methodology", ""),
                )

        # Update thread title
        thread = await sync_to_async(Thread.objects.get)(id=thread_id)
        if not thread.title:
            first_human = await sync_to_async(
                lambda: thread.messages.filter(role="human").first()
            )()
            if first_human:
                thread.title = first_human.content[:100]
                await sync_to_async(thread.save)()

        yield f"event: node/status\ndata: {json.dumps({'node': '__graph__', 'status': 'finished'})}\n\n"

    except Exception as e:
        yield f"event: node/status\ndata: {json.dumps({'node': '__graph__', 'status': 'finished'})}\n\n"
        error_data = {"type": "error", "content": str(e)}
        yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

    yield "event: end\ndata: {}\n\n"


@csrf_exempt
async def stream_run(request, thread_id):
    """SSE endpoint: run LangGraph agent and stream tokens."""
    if request.method != "POST":
        return StreamingHttpResponse(
            "Method not allowed", status=405, content_type="text/plain"
        )

    body = json.loads(request.body)
    input_messages = body.get("input", {}).get("messages", [])
    model = body.get("model", "gpt-4.1-mini")
    lc_messages = parse_input_messages(input_messages)

    # Ensure thread exists
    thread_exists = await sync_to_async(
        Thread.objects.filter(id=thread_id).exists
    )()
    if not thread_exists:
        await sync_to_async(Thread.objects.create)(id=thread_id)

    # Save human message(s) to database
    for msg in lc_messages:
        if isinstance(msg, HumanMessage):
            await sync_to_async(Message.objects.create)(
                thread_id=thread_id,
                role="human",
                content=msg.content,
            )

    run_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": str(thread_id)}}

    async def event_generator():
        async with get_graph_with_checkpointer() as compiled_graph:
            graph_input = {"messages": lc_messages, "model": model}
            async for sse in _stream_graph(compiled_graph, graph_input, config, thread_id, run_id):
                yield sse

    response = StreamingHttpResponse(
        event_generator(),
        content_type="text/event-stream",
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


@csrf_exempt
async def resume_run(request, thread_id):
    """SSE endpoint: resume a graph run after an HITL interrupt."""
    if request.method != "POST":
        return StreamingHttpResponse(
            "Method not allowed", status=405, content_type="text/plain"
        )

    body = json.loads(request.body)
    resume_value = body.get("resume_value")

    run_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": str(thread_id)}}

    async def event_generator():
        async with get_graph_with_checkpointer() as compiled_graph:
            # Use Command(resume=value) to continue past the interrupt
            graph_input = Command(resume=resume_value)
            async for sse in _stream_graph(compiled_graph, graph_input, config, thread_id, run_id):
                yield sse

    response = StreamingHttpResponse(
        event_generator(),
        content_type="text/event-stream",
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response
