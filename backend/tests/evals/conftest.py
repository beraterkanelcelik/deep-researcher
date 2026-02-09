"""Eval-specific fixtures and configuration."""

import json
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--record",
        action="store_true",
        default=False,
        help="Record mode: run real graph and save eval cases (expensive).",
    )


@pytest.fixture
def record_mode(request):
    """Whether we're in record mode (run graph + save) or offline mode (load + grade)."""
    return request.config.getoption("--record")

DATASETS_DIR = Path(__file__).parent / "datasets"


@pytest.fixture
def eval_graph():
    """Compiled graph without checkpointer for eval runs."""
    from chat.graph import build_graph

    return build_graph().compile()


@pytest.fixture
def tool_selection_tasks():
    """Load tool selection eval dataset."""
    with open(DATASETS_DIR / "tool_selection.json") as f:
        return json.load(f)


@pytest.fixture
def conversation_quality_tasks():
    """Load conversation quality eval dataset."""
    with open(DATASETS_DIR / "conversation_quality.json") as f:
        return json.load(f)


@pytest.fixture
def research_quality_tasks():
    """Load research quality eval dataset."""
    with open(DATASETS_DIR / "research_quality.json") as f:
        return json.load(f)


@pytest.fixture
def rag_accuracy_tasks():
    """Load RAG accuracy eval dataset."""
    with open(DATASETS_DIR / "rag_accuracy.json") as f:
        return json.load(f)


@pytest.fixture
def hitl_eval_graph():
    """Compiled graph with MemorySaver checkpointer for HITL interrupt/resume cycles."""
    from langgraph.checkpoint.memory import MemorySaver

    from chat.graph import build_graph

    return build_graph().compile(checkpointer=MemorySaver())


@pytest.fixture
def deep_research_hitl_tasks():
    """Load deep research HITL eval dataset."""
    with open(DATASETS_DIR / "deep_research_hitl.json") as f:
        return json.load(f)
