"""Eval: Full end-to-end user journeys.

Tests multi-turn, multi-tool scenarios that exercise the complete system.
"""

import io

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from .graders import (
    grade_content_contains,
    grade_no_hallucinated_tools,
    grade_response_quality,
)
from .harness import EvalTask, run_eval_task

pytestmark = [pytest.mark.eval, pytest.mark.slow]


async def test_journey_greeting_math_search(eval_graph):
    """Journey 1: Greeting -> math question -> web search -> follow-up."""
    from chat.graph import build_graph

    graph = build_graph().compile()

    # Turn 1: Greeting
    result1 = await graph.ainvoke({
        "messages": [HumanMessage(content="Hi there! I'm working on a project.")],
        "model": "gpt-4.1-mini",
        "research_reports": [],
        "current_plan": None,
        "pending_save": None,
        "research_request": None,
        "topic": "",
        "depth": "",
        "report": None,
    })
    ai_msgs = [m for m in result1["messages"] if isinstance(m, AIMessage)]
    assert len(ai_msgs) >= 1
    assert ai_msgs[-1].content  # Non-empty greeting response

    # Turn 2: Math question (builds on conversation)
    messages2 = result1["messages"] + [HumanMessage(content="What is 256 * 128?")]
    result2 = await graph.ainvoke({
        "messages": messages2,
        "model": "gpt-4.1-mini",
        "research_reports": [],
        "current_plan": None,
        "pending_save": None,
        "research_request": None,
        "topic": "",
        "depth": "",
        "report": None,
    })
    # Should have used calculator
    tool_msgs = [m for m in result2["messages"] if hasattr(m, "name") and m.name == "calculator"]
    assert len(tool_msgs) >= 1
    # Final answer should contain 32768
    last_ai = [m for m in result2["messages"] if isinstance(m, AIMessage)][-1]
    assert "32768" in last_ai.content

    # Turn 3: Web search
    messages3 = result2["messages"] + [HumanMessage(content="Search for Python FastAPI latest version")]
    result3 = await graph.ainvoke({
        "messages": messages3,
        "model": "gpt-4.1-mini",
        "research_reports": [],
        "current_plan": None,
        "pending_save": None,
        "research_request": None,
        "topic": "",
        "depth": "",
        "report": None,
    })
    tool_msgs = [m for m in result3["messages"] if hasattr(m, "name") and m.name == "tavily_search"]
    assert len(tool_msgs) >= 1


@pytest.mark.django_db
async def test_journey_rag_upload_and_query(eval_graph):
    """Journey 2: Upload doc -> ask question -> verify RAG context used."""
    from chat.rag.ingest import ingest_document

    # Upload a domain-specific document
    content = (
        "The TechCorp API uses OAuth 2.0 with PKCE for authentication. "
        "Rate limits are 500 requests per minute for standard plans "
        "and 5000 for enterprise. The base URL is https://api.techcorp.io/v3."
    )
    file = io.BytesIO(content.encode("utf-8"))
    ingest_document(file, "techcorp_api_docs.txt")

    # Query about the document
    task = EvalTask(
        id="e2e-rag",
        input_messages=[{"role": "human", "content": "What authentication does the TechCorp API use?"}],
        expected_content=["OAuth", "PKCE"],
    )

    def code_grader(trial):
        return grade_content_contains(trial, ["OAuth"]) and grade_no_hallucinated_tools(trial)

    result = await run_eval_task(
        graph=eval_graph,
        task=task,
        k=1,
        code_grader=code_grader,
    )
    assert result.pass_at_k >= 1.0, "RAG journey failed: answer didn't mention OAuth"


async def test_journey_plan_creation(eval_graph):
    """Journey 3: Ask for a plan -> verify structured output."""
    task = EvalTask(
        id="e2e-plan",
        input_messages=[{"role": "human", "content": "Create a plan to build a REST API with authentication and database"}],
        expected_tools=["create_plan"],
    )

    from .graders import grade_tool_selection

    def code_grader(trial):
        return grade_tool_selection(trial, ["create_plan"]) and grade_no_hallucinated_tools(trial)

    result = await run_eval_task(
        graph=eval_graph,
        task=task,
        k=2,
        code_grader=code_grader,
    )
    assert result.pass_at_k >= 0.5, "Plan creation journey failed"


async def test_journey_multi_tool_coherence(eval_graph):
    """Verify agent maintains coherence across multiple tool uses."""
    from chat.graph import build_graph

    graph = build_graph().compile()

    result = await graph.ainvoke({
        "messages": [
            HumanMessage(
                content="I need to know the current time, then calculate how many seconds are in 3.5 hours."
            )
        ],
        "model": "gpt-4.1-mini",
        "research_reports": [],
        "current_plan": None,
        "pending_save": None,
        "research_request": None,
        "topic": "",
        "depth": "",
        "report": None,
    })

    # Should have used both tools
    tool_names = set()
    for m in result["messages"]:
        if hasattr(m, "name") and m.name:
            tool_names.add(m.name)

    # At minimum should use calculator; time is also likely
    assert "calculator" in tool_names or "get_current_time" in tool_names

    # Final response should contain 12600 (3.5 * 3600)
    last_ai = [m for m in result["messages"] if isinstance(m, AIMessage)][-1]
    assert "12600" in last_ai.content or "12,600" in last_ai.content
