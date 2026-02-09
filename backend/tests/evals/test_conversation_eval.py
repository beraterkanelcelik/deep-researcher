"""Eval: Response quality, tone, and helpfulness.

Uses model-based graders (gpt-4.1-mini as judge) with task-specific rubrics.
"""

import pytest

from .graders import grade_conversational_tone, grade_response_quality
from .harness import EvalTask, run_eval_task

pytestmark = [pytest.mark.eval, pytest.mark.slow]

K = 3


async def test_conversation_quality_eval(eval_graph, conversation_quality_tasks):
    """Run conversation quality evals with model-based grading."""
    results = []

    for entry in conversation_quality_tasks:
        task = EvalTask(
            id=entry["id"],
            input_messages=[{"role": "human", "content": entry["input"]}],
            grading_rubric=entry["rubric"],
            tags=["conversation"],
        )

        async def model_grader(trial, rubric=entry["rubric"]):
            return await grade_response_quality(trial.transcript, rubric)

        result = await run_eval_task(
            graph=eval_graph,
            task=task,
            k=K,
            model_grader=model_grader,
        )
        results.append(result)

    # Report
    print(f"\n{'='*60}")
    print(f"Conversation Quality Eval Results (k={K})")
    print(f"{'='*60}")
    for r in results:
        score = r.avg_model_score or 0
        status = "PASS" if score >= 0.7 else "FAIL"
        print(f"  {r.task_id}: avg_score={score:.2f}  pass@{K}={r.pass_at_k:.2f}  [{status}]")

    avg_scores = [r.avg_model_score for r in results if r.avg_model_score is not None]
    overall = sum(avg_scores) / len(avg_scores) if avg_scores else 0
    print(f"{'='*60}")
    print(f"  Overall avg score: {overall:.2f}")
    print(f"{'='*60}")

    assert overall >= 0.6, f"Conversation quality too low: {overall:.2f}"


async def test_conversational_tone_eval(eval_graph, conversation_quality_tasks):
    """Eval tone across a sample of tasks."""
    sample = conversation_quality_tasks[:3]
    scores = []

    for entry in sample:
        task = EvalTask(
            id=entry["id"],
            input_messages=[{"role": "human", "content": entry["input"]}],
        )
        result = await run_eval_task(
            graph=eval_graph,
            task=task,
            k=1,
            model_grader=lambda trial: grade_conversational_tone(trial.transcript),
        )
        if result.avg_model_score is not None:
            scores.append(result.avg_model_score)

    avg = sum(scores) / len(scores) if scores else 0
    print(f"\nConversational Tone avg: {avg:.2f}")
    assert avg >= 0.6, f"Tone score too low: {avg:.2f}"
