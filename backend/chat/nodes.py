import json
import os

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from .schemas import HITLOption, HITLPayload, ResearchReport
from .tools import ALL_TOOLS

MODEL_CONFIG = {
    "gpt-4.1-mini": {"model": "gpt-4.1-mini"},
    "gpt-5-nano-high": {"model": "gpt-5-nano", "reasoning_effort": "high"},
    "gpt-5-nano-medium": {"model": "gpt-5-nano", "reasoning_effort": "medium"},
    "gpt-5-nano-low": {"model": "gpt-5-nano", "reasoning_effort": "low"},
    "gpt-5-nano-minimal": {"model": "gpt-5-nano", "reasoning_effort": "minimal"},
}
DEFAULT_MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with advanced research capabilities. "
    "You can:\n"
    "- Answer questions and have conversations\n"
    "- Get the current time and do calculations\n"
    "- Search the web with tavily_search for quick lookups\n"
    "- Create structured task plans with create_plan\n"
    "- Launch deep research investigations with deep_research (spawns parallel searches, produces structured reports)\n"
    "- Save research reports to the database with save_report\n\n"
    "When a user asks for in-depth research on a topic, use the deep_research tool. "
    "For quick factual lookups, use tavily_search. "
    "Be concise and helpful in your responses."
)


def get_llm(model_key: str = DEFAULT_MODEL):
    config = MODEL_CONFIG.get(model_key, MODEL_CONFIG[DEFAULT_MODEL])
    kwargs = {
        "model": config["model"],
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "streaming": True,
    }
    if "reasoning_effort" in config:
        kwargs["reasoning_effort"] = config["reasoning_effort"]
    return ChatOpenAI(**kwargs)


def get_rag_context(state: dict) -> str:
    """Retrieve RAG context from pgvector if available."""
    messages = state.get("messages", [])
    if not messages:
        return ""

    last_message = messages[-1]
    if not hasattr(last_message, "content") or not last_message.content:
        return ""

    query = last_message.content

    try:
        from chat.rag.retriever import retrieve_documents

        docs = retrieve_documents(query, top_k=3)
        if docs:
            context_parts = [
                f"[Source: {doc['filename']}]\n{doc['content']}" for doc in docs
            ]
            return "\n\n---\n\n".join(context_parts)
    except Exception:
        pass

    return ""


def retrieve_node(state: dict) -> dict:
    """RAG retrieval node: query pgvector and inject context."""
    context = get_rag_context(state)

    if context:
        system_msg = SystemMessage(
            content=(
                f"{SYSTEM_PROMPT}\n\n"
                "Use the following context from uploaded documents "
                "to help answer the user's question. If the context is not relevant, you can ignore it.\n\n"
                f"Context:\n{context}"
            )
        )
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            return {"messages": [system_msg] + list(messages)}

    return {}


def agent_node(state: dict) -> dict:
    """Call LLM with all tools (including sentinel tools for routing)."""
    model = state.get("model", DEFAULT_MODEL)
    llm = get_llm(model)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    messages = state["messages"]

    has_system = any(
        isinstance(m, SystemMessage)
        or (hasattr(m, "type") and m.type == "system")
        for m in messages
    )
    if not has_system:
        system_msg = SystemMessage(content=SYSTEM_PROMPT)
        messages = [system_msg] + list(messages)

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: dict) -> str:
    """Route based on tool calls: detect sentinel tools for special routing."""
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = {tc["name"] for tc in last_message.tool_calls}

        if "deep_research" in tool_names:
            return "deep_research"
        if "save_report" in tool_names:
            return "save_confirm"

        return "tools"

    return "end"


def save_confirm_node(state: dict) -> dict:
    """HITL interrupt #3: confirm saving a research report to DB."""
    reports = state.get("research_reports", [])
    messages = state.get("messages", [])

    # Extract report_index from the save_report tool call
    report_index = 0
    last_message = messages[-1] if messages else None
    if hasattr(last_message, "tool_calls"):
        for tc in last_message.tool_calls:
            if tc["name"] == "save_report":
                report_index = tc["args"].get("report_index", 0)
                break

    if not reports:
        # No reports available - return a ToolMessage indicating this
        tool_call_id = ""
        if hasattr(last_message, "tool_calls"):
            for tc in last_message.tool_calls:
                if tc["name"] == "save_report":
                    tool_call_id = tc["id"]
                    break
        return {
            "messages": [
                ToolMessage(
                    content="No research reports available to save. Run a deep research first.",
                    tool_call_id=tool_call_id,
                    name="save_report",
                )
            ],
            "pending_save": None,
        }

    # Get the report at the requested index (0 = most recent)
    idx = min(report_index, len(reports) - 1)
    report = reports[-(idx + 1)]  # Reverse index for most recent first

    payload = HITLPayload(
        hitl_type="confirm",
        title="Save Research Report",
        message="Do you want to save this research report to the database?",
        report=ResearchReport(**report) if isinstance(report, dict) else report,
        options=[
            HITLOption(id="save", label="Save", description="Save report to database"),
            HITLOption(id="cancel", label="Cancel", description="Don't save"),
        ],
    )

    user_response = interrupt(payload.model_dump())

    action = user_response.get("action", "cancel") if isinstance(user_response, dict) else str(user_response)

    if action == "save":
        return {"pending_save": report}
    else:
        # User cancelled - respond with ToolMessage
        tool_call_id = ""
        if hasattr(last_message, "tool_calls"):
            for tc in last_message.tool_calls:
                if tc["name"] == "save_report":
                    tool_call_id = tc["id"]
                    break
        return {
            "messages": [
                ToolMessage(
                    content="Report save cancelled by user.",
                    tool_call_id=tool_call_id,
                    name="save_report",
                )
            ],
            "pending_save": None,
        }


def save_to_db_node(state: dict) -> dict:
    """Persist the pending report to the Django ResearchReport model."""
    pending = state.get("pending_save")
    messages = state.get("messages", [])

    # Find the tool_call_id for the save_report call
    tool_call_id = ""
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                if tc["name"] == "save_report":
                    tool_call_id = tc["id"]
                    break
            if tool_call_id:
                break

    if not pending:
        return {
            "messages": [
                ToolMessage(
                    content="No report to save.",
                    tool_call_id=tool_call_id,
                    name="save_report",
                )
            ],
            "pending_save": None,
        }

    # The actual DB save happens in views_stream.py after the graph completes,
    # because we need the Django ORM context and thread_id.
    # Here we just confirm the intent and format the response.
    report_title = pending.get("title", "Untitled Report") if isinstance(pending, dict) else "Report"

    return {
        "messages": [
            ToolMessage(
                content=json.dumps({
                    "status": "saved",
                    "title": report_title,
                    "message": f"Research report '{report_title}' has been saved to the database.",
                }),
                tool_call_id=tool_call_id,
                name="save_report",
            )
        ],
    }


def should_continue_after_save_confirm(state: dict) -> str:
    """Route after save confirmation: to save_to_db if pending, else back to agent."""
    if state.get("pending_save"):
        return "save_to_db"
    return "agent"


def prepare_research_node(state: dict) -> dict:
    """Extract topic, depth, and tool_call_id from the agent's deep_research tool call."""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    topic = ""
    depth = "standard"
    tool_call_id = ""

    if hasattr(last_message, "tool_calls"):
        for tc in last_message.tool_calls:
            if tc["name"] == "deep_research":
                topic = tc["args"].get("topic", "")
                depth = tc["args"].get("depth", "standard")
                tool_call_id = tc["id"]
                break

    if not topic:
        return {
            "messages": [
                ToolMessage(
                    content="No research topic provided.",
                    tool_call_id=tool_call_id or "unknown",
                    name="deep_research",
                )
            ],
            "topic": "",
            "depth": "standard",
        }

    return {"topic": topic, "depth": depth}


def process_research_result_node(state: dict) -> dict:
    """Read the subgraph's report output and format it as a ToolMessage for the agent."""
    report = state.get("report")
    messages = state.get("messages", [])

    # Find tool_call_id for the deep_research call
    tool_call_id = ""
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                if tc["name"] == "deep_research":
                    tool_call_id = tc["id"]
                    break
            if tool_call_id:
                break

    response_parts = []
    if report:
        response_parts.append(
            f"**Research Report: {report.get('title', 'Untitled')}**\n\n"
            f"{report.get('summary', 'No summary available.')}\n\n"
            f"**Key Findings:**\n"
        )
        for i, finding in enumerate(report.get("key_findings", []), 1):
            insight = finding.get("insight", "") if isinstance(finding, dict) else str(finding)
            response_parts.append(f"{i}. {insight}")

        sources = report.get("sources", [])
        if sources:
            response_parts.append(f"\n\n**Sources:** {len(sources)} sources referenced")
    else:
        response_parts.append("Research completed but no report was generated.")

    content = "\n".join(response_parts)

    result = {
        "messages": [
            ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
                name="deep_research",
            )
        ],
    }
    if report:
        result["research_reports"] = [report]

    return result


def should_run_research(state: dict) -> str:
    """Route after prepare_research: run subgraph if topic is set, else skip back to agent."""
    if state.get("topic"):
        return "deep_research"
    return "agent"
