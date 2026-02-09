import operator
import os
from contextlib import asynccontextmanager
from typing import Annotated

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from .nodes import (
    agent_node,
    prepare_research_node,
    process_research_result_node,
    retrieve_node,
    save_confirm_node,
    save_to_db_node,
    should_continue,
    should_continue_after_save_confirm,
    should_run_research,
)
from .research_graph import build_research_subgraph
from .tools import SIMPLE_TOOLS


class AgentState(MessagesState):
    """Extended state with messages key (required by assistant-ui) plus research fields."""

    model: str
    research_reports: Annotated[list, operator.add]
    current_plan: dict | None
    pending_save: dict | None
    research_request: dict | None
    # Shared keys with research subgraph (ResearchInput / ResearchOutput)
    topic: str
    depth: str
    report: dict | None


def build_graph():
    """Build the LangGraph agent graph with research subgraph as a native node."""
    graph = StateGraph(AgentState)

    # Compile the research subgraph (no checkpointer — parent's propagates automatically)
    compiled_research = build_research_subgraph().compile()

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools=SIMPLE_TOOLS))
    graph.add_node("prepare_research", prepare_research_node)
    graph.add_node("deep_research", compiled_research)
    graph.add_node("process_research_result", process_research_result_node)
    graph.add_node("save_confirm", save_confirm_node)
    graph.add_node("save_to_db", save_to_db_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "deep_research": "prepare_research",
            "save_confirm": "save_confirm",
            "end": END,
        },
    )

    graph.add_edge("tools", "agent")

    # prepare_research -> conditional: run subgraph or skip back to agent on error
    graph.add_conditional_edges(
        "prepare_research",
        should_run_research,
        {
            "deep_research": "deep_research",
            "agent": "agent",
        },
    )

    graph.add_edge("deep_research", "process_research_result")
    graph.add_edge("process_research_result", "agent")

    graph.add_conditional_edges(
        "save_confirm",
        should_continue_after_save_confirm,
        {
            "save_to_db": "save_to_db",
            "agent": "agent",
        },
    )
    graph.add_edge("save_to_db", "agent")

    return graph


def get_graph():
    """Get compiled graph without checkpointer (for simple usage)."""
    graph = build_graph()
    return graph.compile()


@asynccontextmanager
async def get_graph_with_checkpointer():
    """Get compiled graph with PostgreSQL checkpointer (async context manager)."""
    database_url = os.environ.get(
        "DATABASE_URL", "postgresql://chat:chat@localhost:5432/chatdb"
    )

    async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
        await checkpointer.setup()
        graph = build_graph()
        yield graph.compile(checkpointer=checkpointer)
