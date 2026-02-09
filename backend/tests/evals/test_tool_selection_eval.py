"""Eval: Agent selects correct tools for given inputs.

Runs each task k=3 times, applies code graders (tool selection, no hallucinated tools),
and reports pass@k and pass^k metrics.
"""

import pytest

from .graders import grade_no_hallucinated_tools, grade_tool_selection
from .harness import EvalTask, run_eval_task

pytestmark = [pytest.mark.eval, pytest.mark.slow]

K = 3


@pytest.fixture
def make_eval_task():
    """Convert raw dataset entry to EvalTask."""

    def _make(entry):
        return EvalTask(
            id=entry["id"],
            input_messages=[{"role": "human", "content": entry["input"]}],
            expected_tools=entry["expected_tools"],
            tags=entry.get("tags", []),
        )

    return _make


async def test_tool_selection_eval(eval_graph, tool_selection_tasks, make_eval_task):
    """Run tool selection evals across all tasks."""
    results = []

    for entry in tool_selection_tasks:
        task = make_eval_task(entry)

        def code_grader(trial, expected=task.expected_tools):
            return grade_tool_selection(trial, expected) and grade_no_hallucinated_tools(trial)

        result = await run_eval_task(
            graph=eval_graph,
            task=task,
            k=K,
            code_grader=code_grader,
        )
        results.append(result)

    # Report metrics
    total_pass_at_k = sum(r.pass_at_k for r in results) / len(results)
    total_pass_pow_k = sum(r.pass_pow_k for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"Tool Selection Eval Results (k={K})")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r.pass_at_k >= 1.0 else "PARTIAL" if r.pass_at_k > 0 else "FAIL"
        print(f"  {r.task_id}: pass@{K}={r.pass_at_k:.2f}  pass^{K}={r.pass_pow_k:.2f}  [{status}]")
    print(f"{'='*60}")
    print(f"  Aggregate pass@{K}: {total_pass_at_k:.2f}")
    print(f"  Aggregate pass^{K}: {total_pass_pow_k:.2f}")
    print(f"{'='*60}")

    # Soft assertion: expect at least 70% pass@k
    assert total_pass_at_k >= 0.7, f"Tool selection pass@{K} too low: {total_pass_at_k:.2f}"


async def test_no_hallucinated_tools(eval_graph, tool_selection_tasks, make_eval_task):
    """Verify agent never calls non-existent tools."""
    for entry in tool_selection_tasks[:5]:  # Sample for speed
        task = make_eval_task(entry)
        result = await run_eval_task(
            graph=eval_graph,
            task=task,
            k=1,
            code_grader=grade_no_hallucinated_tools,
        )
        assert result.pass_at_k == 1.0, f"Task {task.id} used hallucinated tools"
