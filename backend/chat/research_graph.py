"""Deep research subgraph with HITL interrupts and parallel explorer nodes."""

from __future__ import annotations

import operator
import os
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .schemas import (
    ExplorerInstruction,
    HITLOption,
    HITLPayload,
    InstructionList,
    ResearchReport,
    SearchResult,
    SubTopicList,
)


# --- State definitions ---


class ResearchInput(TypedDict):
    """Input schema for the deep research subgraph (parent provides these)."""

    topic: str
    depth: str


class ResearchOutput(TypedDict):
    """Output schema for the deep research subgraph (parent receives these)."""

    report: dict | None


class DeepResearchState(TypedDict):
    """Full internal state for the deep research subgraph."""

    topic: str
    depth: str
    clarified_topics: list[str]
    explorer_instructions: list[dict]
    search_results: Annotated[list[dict], operator.add]
    report: dict | None
    status: str


class ExplorerState(TypedDict):
    """State for a single parallel explorer node."""

    query: str
    search_focus: str
    context: str
    search_results: Annotated[list[dict], operator.add]


# --- Nodes ---

DEPTH_MAP = {"quick": 3, "standard": 5, "deep": 8}


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )


def clarify_node(state: dict) -> dict:
    """LLM suggests sub-topics, then interrupt for user to select via checkboxes."""
    topic = state.get("topic", "")
    depth = state.get("depth", "standard")

    llm = _get_llm()
    structured_llm = llm.with_structured_output(SubTopicList)
    result = structured_llm.invoke(
        f"Given the research topic: '{topic}'\n\n"
        f"Suggest 5-8 specific sub-topics or angles to investigate."
    )
    subtopics = result.topics

    options = [
        HITLOption(id=f"topic_{i}", label=t, description="", selected=True)
        for i, t in enumerate(subtopics)
    ]

    payload = HITLPayload(
        hitl_type="checkbox",
        title="Select Research Topics",
        message=f"The following sub-topics were identified for '{topic}'. Select the ones you'd like to research:",
        options=options,
    )

    # HITL interrupt #1: user selects sub-topics
    user_response = interrupt(payload.model_dump())

    # Parse user selection
    selected_ids = user_response if isinstance(user_response, list) else []
    selected_topics = []
    for opt in options:
        if opt.id in selected_ids:
            selected_topics.append(opt.label)

    # If nothing selected, use all
    if not selected_topics:
        selected_topics = subtopics

    return {
        "clarified_topics": selected_topics,
        "status": "clarified",
    }


def orchestrate_node(state: dict) -> dict:
    """Generate N explorer instructions using structured output."""
    topics = state.get("clarified_topics", [])
    depth = state.get("depth", "standard")
    num_explorers = min(len(topics), DEPTH_MAP.get(depth, 5))

    llm = _get_llm()
    structured_llm = llm.with_structured_output(InstructionList)

    result = structured_llm.invoke(
        f"You are creating search instructions for {num_explorers} parallel web researchers.\n\n"
        f"Research topics to cover:\n"
        + "\n".join(f"- {t}" for t in topics)
        + f"\n\nCreate exactly {num_explorers} search instructions. Each should have:\n"
        "- A specific, well-crafted search query for Tavily web search\n"
        "- A clear search focus describing what to look for\n"
        "- Optional context for the researcher"
    )

    instructions = [inst.model_dump() for inst in result.instructions[:num_explorers]]
    return {
        "explorer_instructions": instructions,
        "status": "orchestrated",
    }


def route_to_explorers(state: dict) -> list[Send]:
    """Conditional edge that fans out to N parallel explorer nodes via Send()."""
    instructions = state.get("explorer_instructions", [])
    return [
        Send(
            "explorer",
            {
                "query": inst["query"],
                "search_focus": inst["search_focus"],
                "context": inst.get("context", ""),
                "search_results": [],
            },
        )
        for inst in instructions
    ]


def explorer_node(state: dict) -> dict:
    """Run Tavily search and return results."""
    query = state.get("query", "")
    search_focus = state.get("search_focus", "")

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        search = TavilySearchResults(max_results=3)
        raw_results = search.invoke({"query": query})

        results = []
        if isinstance(raw_results, list):
            for r in raw_results:
                results.append(
                    SearchResult(
                        url=r.get("url", ""),
                        title=r.get("title", query),
                        content=r.get("content", ""),
                        score=r.get("score", 0.0),
                    ).model_dump()
                )
        elif isinstance(raw_results, str):
            results.append(
                SearchResult(title=query, content=raw_results).model_dump()
            )
    except Exception as e:
        results = [
            SearchResult(title=query, content=f"Search error: {e}").model_dump()
        ]

    return {"search_results": results}


def synthesize_node(state: dict) -> dict:
    """Merge all search results into a structured ResearchReport."""
    topic = state.get("topic", "")
    search_results = state.get("search_results", [])
    clarified_topics = state.get("clarified_topics", [])

    results_text = ""
    for i, r in enumerate(search_results, 1):
        results_text += (
            f"\n--- Result {i} ---\n"
            f"Title: {r.get('title', 'N/A')}\n"
            f"URL: {r.get('url', 'N/A')}\n"
            f"Content: {r.get('content', 'N/A')}\n"
        )

    llm = _get_llm()
    structured_llm = llm.with_structured_output(ResearchReport)

    report = structured_llm.invoke(
        f"Synthesize the following search results into a comprehensive research report.\n\n"
        f"Original topic: {topic}\n"
        f"Sub-topics investigated: {', '.join(clarified_topics)}\n\n"
        f"Search Results:\n{results_text}\n\n"
        "Create a well-structured report with:\n"
        "- A clear title\n"
        "- Executive summary\n"
        "- Key findings with evidence and source URLs\n"
        "- List of all source URLs\n"
        "- Relevant tags\n"
        "- Brief methodology description"
    )

    return {
        "report": report.model_dump(),
        "status": "synthesized",
    }


def review_node(state: dict) -> dict:
    """HITL interrupt #2: user reviews the report (approve/edit/redo)."""
    report = state.get("report")

    payload = HITLPayload(
        hitl_type="review",
        title="Review Research Report",
        message="Please review the research report below. You can approve it, edit it, or request a redo.",
        report=ResearchReport(**report) if report else None,
        options=[
            HITLOption(id="approve", label="Approve", description="Accept this report"),
            HITLOption(id="edit", label="Edit", description="Modify the report"),
            HITLOption(id="redo", label="Redo", description="Re-run the research"),
        ],
    )

    user_response = interrupt(payload.model_dump())

    action = user_response.get("action", "approve") if isinstance(user_response, dict) else "approve"

    if action == "edit" and isinstance(user_response, dict):
        edits = user_response.get("edits", {})
        if report and edits:
            report.update(edits)
            return {"report": report, "status": "approved"}

    if action == "redo":
        return {"status": "redo"}

    return {"status": "approved"}


def should_continue_review(state: dict) -> str:
    """Route after review: END if approved, back to orchestrate for redo."""
    if state.get("status") == "redo":
        return "orchestrate"
    return END


# --- Build subgraph ---


def build_research_subgraph() -> StateGraph:
    """Build and return the deep research subgraph (uncompiled)."""
    graph = StateGraph(DeepResearchState, input=ResearchInput, output=ResearchOutput)

    graph.add_node("clarify", clarify_node)
    graph.add_node("orchestrate", orchestrate_node)
    graph.add_node("explorer", explorer_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("review", review_node)

    graph.add_edge(START, "clarify")
    graph.add_edge("clarify", "orchestrate")
    graph.add_conditional_edges("orchestrate", route_to_explorers, ["explorer"])
    graph.add_edge("explorer", "synthesize")
    graph.add_edge("synthesize", "review")
    graph.add_conditional_edges(
        "review",
        should_continue_review,
        {
            "orchestrate": "orchestrate",
            END: END,
        },
    )

    return graph
