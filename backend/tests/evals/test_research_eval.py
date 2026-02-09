"""Eval: Research report quality — groundedness, coverage, and structure.

Runs the research subgraph directly (bypasses HITL with mock inputs).
"""

import pytest

from chat.research_graph import explorer_node, orchestrate_node, synthesize_node
from .graders import grade_report_structure, grade_research_coverage, grade_research_groundedness

pytestmark = [pytest.mark.eval, pytest.mark.slow]


async def test_research_report_structure(research_quality_tasks):
    """Verify synthesized reports have correct structure."""
    # Use a subset for speed
    for entry in research_quality_tasks[:3]:
        # Simulate orchestrate -> explore -> synthesize pipeline
        orchestrate_state = {
            "clarified_topics": [entry["topic"]],
            "depth": "quick",
        }
        orch_result = orchestrate_node(orchestrate_state)

        # Run one explorer
        all_results = []
        for inst in orch_result["explorer_instructions"][:2]:
            exp_result = explorer_node({
                "query": inst["query"],
                "search_focus": inst["search_focus"],
                "context": inst.get("context", ""),
                "search_results": [],
            })
            all_results.extend(exp_result["search_results"])

        # Synthesize
        synth_state = {
            "topic": entry["topic"],
            "clarified_topics": [entry["topic"]],
            "search_results": all_results,
        }
        synth_result = synthesize_node(synth_state)
        report = synth_result["report"]

        assert grade_report_structure(report), f"Report structure invalid for {entry['id']}"


async def test_research_groundedness(research_quality_tasks):
    """Verify reports are grounded in their sources."""
    entry = research_quality_tasks[0]

    orchestrate_state = {
        "clarified_topics": [entry["topic"]],
        "depth": "quick",
    }
    orch_result = orchestrate_node(orchestrate_state)

    all_results = []
    for inst in orch_result["explorer_instructions"][:2]:
        exp_result = explorer_node({
            "query": inst["query"],
            "search_focus": inst["search_focus"],
            "context": inst.get("context", ""),
            "search_results": [],
        })
        all_results.extend(exp_result["search_results"])

    synth_state = {
        "topic": entry["topic"],
        "clarified_topics": [entry["topic"]],
        "search_results": all_results,
    }
    synth_result = synthesize_node(synth_state)
    report = synth_result["report"]

    score = await grade_research_groundedness(report, all_results)
    print(f"\nGroundedness score for '{entry['topic']}': {score:.2f}")
    assert score >= 0.5, f"Groundedness too low: {score:.2f}"


async def test_research_coverage(research_quality_tasks):
    """Verify reports cover key aspects of the topic."""
    entry = research_quality_tasks[0]

    orchestrate_state = {
        "clarified_topics": [entry["topic"]],
        "depth": "quick",
    }
    orch_result = orchestrate_node(orchestrate_state)

    all_results = []
    for inst in orch_result["explorer_instructions"][:2]:
        exp_result = explorer_node({
            "query": inst["query"],
            "search_focus": inst["search_focus"],
            "context": inst.get("context", ""),
            "search_results": [],
        })
        all_results.extend(exp_result["search_results"])

    synth_state = {
        "topic": entry["topic"],
        "clarified_topics": [entry["topic"]],
        "search_results": all_results,
    }
    synth_result = synthesize_node(synth_state)
    report = synth_result["report"]

    score = await grade_research_coverage(report, entry["topic"])
    print(f"\nCoverage score for '{entry['topic']}': {score:.2f}")
    assert score >= 0.5, f"Coverage too low: {score:.2f}"
