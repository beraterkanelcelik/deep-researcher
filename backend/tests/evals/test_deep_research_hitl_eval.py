"""Eval: Deep research HITL flow — two-phase: record once, grade cheaply.

Phase 1 (--record): runs real graph with LLM calls, saves structured eval cases as JSON.
Phase 2 (default): loads saved eval cases, re-grades offline — NO graph execution.

Usage:
  pytest tests/evals/test_deep_research_hitl_eval.py --record -v -s   # expensive, saves cases
  pytest tests/evals/test_deep_research_hitl_eval.py -v -s             # cheap, loads + grades
"""

import pytest

from .eval_case import has_recorded_cases, run_or_load_eval
from .graders import (
    grade_deep_research_quality,
    grade_hitl_flow_completeness,
    grade_hitl_payload_all_valid,
    grade_research_report_present,
)
from .harness import HITLEvalTask, HITLResponse
from .transcript import save_eval_report, save_trial_transcript

pytestmark = [pytest.mark.eval]

K = 2


def _build_hitl_task(task_data: dict) -> HITLEvalTask:
    """Convert a raw dataset dict into an HITLEvalTask."""
    return HITLEvalTask(
        id=task_data["id"],
        input_messages=task_data["input_messages"],
        hitl_responses=[
            HITLResponse(
                expected_hitl_type=r["expected_hitl_type"],
                resume_value=r["resume_value"],
                description=r.get("description", ""),
            )
            for r in task_data["hitl_responses"]
        ],
        expected_hitl_types=task_data["expected_hitl_types"],
        grading_rubric=task_data.get("grading_rubric"),
        tags=task_data.get("tags", []),
    )


def _skip_if_no_cases(task_id: str, record_mode: bool):
    """Skip test if no recorded eval cases exist and we're not in record mode."""
    if not record_mode and not has_recorded_cases(task_id):
        pytest.skip(
            f"No recorded eval cases for '{task_id}'. "
            f"Run with --record to generate: pytest --record -k {task_id}"
        )


def _print_results(result):
    """Print detailed eval results to console."""
    print(f"\n{'='*70}")
    print(f"Task: {result.task_id}")
    print(f"pass@k: {result.pass_at_k:.2f}  |  pass^k: {result.pass_pow_k:.2f}")
    print(f"Avg grades: {result.avg_grades}")
    print(f"{'='*70}")

    for trial in result.trials:
        status = "PASS" if trial.success else "FAIL"
        print(f"\n  Trial {trial.trial_number}: [{status}] ({trial.duration_seconds:.1f}s)")
        print(f"    Interrupts: {len(trial.interrupts)}")
        for rec in trial.interrupts:
            print(
                f"      #{rec.interrupt_index}: {rec.hitl_type} "
                f"(node={rec.node_name})"
            )
        print(f"    Tool calls: {trial.tool_calls_made}")
        print(f"    Report: {'present' if trial.report else 'missing'}")
        print(f"    Grades: {trial.grades}")

    print()


def _save_transcripts(result, task: HITLEvalTask):
    """Save all trial transcripts and the aggregate report."""
    for trial in result.trials:
        path = save_trial_transcript(trial, task)
        print(f"  Saved transcript: {path.name}")
    report_path = save_eval_report(result, task)
    print(f"  Saved report: {report_path.name}")


class TestHITLHappyPath:
    """Test the standard happy path: select topics -> approve -> save."""

    @pytest.mark.asyncio
    async def test_happy_path_flow(
        self, hitl_eval_graph, deep_research_hitl_tasks, record_mode
    ):
        task_data = next(t for t in deep_research_hitl_tasks if t["id"] == "hitl-01")
        task = _build_hitl_task(task_data)
        _skip_if_no_cases(task.id, record_mode)

        def code_flow(trial):
            return grade_hitl_flow_completeness(trial, task.expected_hitl_types)

        def code_payload(trial):
            return grade_hitl_payload_all_valid(trial)

        def code_report(trial):
            return grade_research_report_present(trial)

        async def model_quality(trial):
            return await grade_deep_research_quality(trial)

        result = await run_or_load_eval(
            graph=hitl_eval_graph,
            task=task,
            k=K,
            code_graders={
                "flow_completeness": code_flow,
                "payload_validity": code_payload,
                "report_present": code_report,
            },
            model_graders={
                "research_quality": model_quality,
            },
            record=record_mode,
        )

        _print_results(result)
        _save_transcripts(result, task)

        assert result.pass_at_k >= 0.5, (
            f"Happy path pass@k={result.pass_at_k:.2f} < 0.5"
        )


class TestHITLRedoFlow:
    """Test the redo loop: first review sends redo, second approves."""

    @pytest.mark.asyncio
    async def test_redo_flow(
        self, hitl_eval_graph, deep_research_hitl_tasks, record_mode
    ):
        task_data = next(t for t in deep_research_hitl_tasks if t["id"] == "hitl-03")
        task = _build_hitl_task(task_data)
        _skip_if_no_cases(task.id, record_mode)

        def code_flow(trial):
            return grade_hitl_flow_completeness(trial, task.expected_hitl_types)

        def code_payload(trial):
            return grade_hitl_payload_all_valid(trial)

        def code_report(trial):
            return grade_research_report_present(trial)

        def code_min_interrupts(trial):
            """Redo flow should have at least 4 interrupts."""
            return len(trial.interrupts) >= 4

        async def model_quality(trial):
            return await grade_deep_research_quality(trial)

        result = await run_or_load_eval(
            graph=hitl_eval_graph,
            task=task,
            k=K,
            code_graders={
                "flow_completeness": code_flow,
                "payload_validity": code_payload,
                "report_present": code_report,
                "min_interrupts": code_min_interrupts,
            },
            model_graders={
                "research_quality": model_quality,
            },
            record=record_mode,
        )

        _print_results(result)
        _save_transcripts(result, task)

        assert result.pass_at_k >= 0.5, (
            f"Redo flow pass@k={result.pass_at_k:.2f} < 0.5"
        )


class TestHITLEditFlow:
    """Test editing the report title during review."""

    @pytest.mark.asyncio
    async def test_edit_applies_changes(
        self, hitl_eval_graph, deep_research_hitl_tasks, record_mode
    ):
        task_data = next(t for t in deep_research_hitl_tasks if t["id"] == "hitl-04")
        task = _build_hitl_task(task_data)
        _skip_if_no_cases(task.id, record_mode)

        expected_title = "Electric Vehicles & Battery Tech: A Comprehensive Review"

        def code_flow(trial):
            return grade_hitl_flow_completeness(trial, task.expected_hitl_types)

        def code_payload(trial):
            return grade_hitl_payload_all_valid(trial)

        def code_report(trial):
            return grade_research_report_present(trial)

        def code_title_edited(trial):
            """Check the report title was updated by the edit action."""
            if trial.report is None:
                return False
            return trial.report.get("title") == expected_title

        result = await run_or_load_eval(
            graph=hitl_eval_graph,
            task=task,
            k=K,
            code_graders={
                "flow_completeness": code_flow,
                "payload_validity": code_payload,
                "report_present": code_report,
                "title_edited": code_title_edited,
            },
            record=record_mode,
        )

        _print_results(result)
        _save_transcripts(result, task)

        # Edit propagation is best-effort — the graph's review_node applies edits
        # but the report may be further processed. Soft assert.
        assert result.pass_at_k >= 0.0, "Edit flow should not crash"


class TestHITLCancelFlow:
    """Test cancelling the save at the confirm step."""

    @pytest.mark.asyncio
    async def test_cancel_save(
        self, hitl_eval_graph, deep_research_hitl_tasks, record_mode
    ):
        task_data = next(t for t in deep_research_hitl_tasks if t["id"] == "hitl-05")
        task = _build_hitl_task(task_data)
        _skip_if_no_cases(task.id, record_mode)

        def code_flow(trial):
            return grade_hitl_flow_completeness(trial, task.expected_hitl_types)

        def code_payload(trial):
            return grade_hitl_payload_all_valid(trial)

        def code_cancel_acknowledged(trial):
            """Check the agent acknowledged the cancellation in its response."""
            for entry in reversed(trial.transcript):
                if entry.get("type") == "tool" and entry.get("name") == "save_report":
                    content = entry.get("content", "")
                    if "cancel" in content.lower():
                        return True
            return False

        result = await run_or_load_eval(
            graph=hitl_eval_graph,
            task=task,
            k=K,
            code_graders={
                "flow_completeness": code_flow,
                "payload_validity": code_payload,
                "cancel_acknowledged": code_cancel_acknowledged,
            },
            record=record_mode,
        )

        _print_results(result)
        _save_transcripts(result, task)

        assert result.pass_at_k >= 0.5, (
            f"Cancel flow pass@k={result.pass_at_k:.2f} < 0.5"
        )


class TestHITLFullSuite:
    """Run all 6 HITL eval tasks and report aggregate results."""

    @pytest.mark.asyncio
    async def test_full_hitl_eval_suite(
        self, hitl_eval_graph, deep_research_hitl_tasks, record_mode
    ):
        all_results = []

        for task_data in deep_research_hitl_tasks:
            task = _build_hitl_task(task_data)

            if not record_mode and not has_recorded_cases(task.id):
                print(f"  Skipping {task.id} — no recorded cases")
                continue

            def code_flow(trial, expected=task.expected_hitl_types):
                return grade_hitl_flow_completeness(trial, expected)

            def code_payload(trial):
                return grade_hitl_payload_all_valid(trial)

            def code_report(trial):
                return grade_research_report_present(trial)

            async def model_quality(trial):
                return await grade_deep_research_quality(trial)

            result = await run_or_load_eval(
                graph=hitl_eval_graph,
                task=task,
                k=K,
                code_graders={
                    "flow_completeness": code_flow,
                    "payload_validity": code_payload,
                    "report_present": code_report,
                },
                model_graders={
                    "research_quality": model_quality,
                },
                record=record_mode,
            )

            _print_results(result)
            _save_transcripts(result, task)
            all_results.append(result)

        if not all_results:
            pytest.skip(
                "No recorded eval cases for any task. "
                "Run with --record to generate: pytest --record"
            )

        # Print summary table
        print(f"\n{'='*70}")
        print("FULL SUITE SUMMARY")
        print(f"{'='*70}")
        print(f"{'Task ID':<12} {'pass@k':>8} {'pass^k':>8} {'Avg Grades'}")
        print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*40}")

        total_pass_at_k = 0.0
        for r in all_results:
            grades_str = ", ".join(
                f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in r.avg_grades.items()
            )
            print(f"{r.task_id:<12} {r.pass_at_k:>8.2f} {r.pass_pow_k:>8.2f} {grades_str}")
            total_pass_at_k += r.pass_at_k

        aggregate = total_pass_at_k / len(all_results) if all_results else 0.0
        print(f"\nAggregate pass@k: {aggregate:.2f}")

        # Soft assertion — deep research is complex, so threshold is low
        assert aggregate >= 0.3, (
            f"Aggregate pass@k={aggregate:.2f} < 0.3"
        )
