"""Two-phase eval: record expensive graph runs once, grade offline cheaply.

Implements the Anthropic "Demystifying Evals" pattern:
  input -> expected behavior -> actual output -> grading

Usage:
  pytest --record   # Phase 1: run real graph, save eval cases (expensive)
  pytest            # Phase 2: load saved cases, re-grade (cheap, no graph)

GenericFakeChatModel from langchain_core can be used to simulate LLM
responses for fully offline recording (patch chat.research_graph._get_llm
and chat.nodes.get_llm to return fake models).
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .harness import (
    HITLEvalResult,
    HITLEvalTask,
    HITLInterruptRecord,
    HITLTrial,
    run_hitl_eval_task,
)
from .metrics import pass_at_k, pass_pow_k
from .transcript import _serialize_grades, _serialize_value

RECORDED_CASES_DIR = Path(__file__).parent / "recorded_cases"


# ── Eval Case JSON Schema ─────────────────────────────────────────


def _build_eval_case(
    trial: HITLTrial,
    task: HITLEvalTask,
    model: str = "gpt-4.1-mini",
) -> dict:
    """Build a structured eval case dict from a completed trial.

    Follows the Anthropic eval structure: meta / input / expected / actual / grades.
    """
    return {
        "meta": {
            "task_id": task.id,
            "trial_number": trial.trial_number,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "tags": task.tags,
        },
        "input": {
            "messages": task.input_messages,
            "hitl_responses": [
                {
                    "expected_hitl_type": r.expected_hitl_type,
                    "resume_value": _serialize_value(r.resume_value),
                    "description": r.description,
                }
                for r in task.hitl_responses
            ],
        },
        "expected": {
            "hitl_types": task.expected_hitl_types,
            "grading_rubric": task.grading_rubric,
        },
        "actual": {
            "transcript": _serialize_value(trial.transcript),
            "interrupts": [
                {
                    "interrupt_index": rec.interrupt_index,
                    "node_name": rec.node_name,
                    "hitl_type": rec.hitl_type,
                    "payload": _serialize_value(rec.payload),
                    "resume_value": _serialize_value(rec.resume_value),
                    "timestamp": rec.timestamp,
                }
                for rec in trial.interrupts
            ],
            "tool_calls_made": trial.tool_calls_made,
            "report": _serialize_value(trial.report),
            "outcome": _serialize_value(trial.outcome),
            "duration_seconds": round(trial.duration_seconds, 2),
        },
        "grades": {
            "code": _serialize_grades(
                {k: v for k, v in trial.grades.items() if isinstance(v, bool)}
            ),
            "model": _serialize_grades(
                {
                    k: v
                    for k, v in trial.grades.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
            ),
        },
    }


# ── Record Functions ───────────────────────────────────────────────


def save_eval_case(
    trial: HITLTrial,
    task: HITLEvalTask,
    model: str = "gpt-4.1-mini",
) -> Path:
    """Save a single trial as a structured eval case JSON file.

    Uses deterministic filenames so re-recording overwrites cleanly.
    """
    RECORDED_CASES_DIR.mkdir(parents=True, exist_ok=True)

    case = _build_eval_case(trial, task, model)

    filename = f"{task.id}_trial{trial.trial_number}.json"
    filepath = RECORDED_CASES_DIR / filename
    filepath.write_text(json.dumps(case, indent=2, ensure_ascii=False))
    return filepath


async def record_eval_cases(
    graph,
    task: HITLEvalTask,
    k: int = 2,
    code_graders: dict | None = None,
    model_graders: dict | None = None,
    model: str = "gpt-4.1-mini",
) -> HITLEvalResult:
    """Phase 1 (Record mode): run real graph and save each trial as an eval case.

    Returns the HITLEvalResult from the live run.
    """
    result = await run_hitl_eval_task(
        graph=graph,
        task=task,
        k=k,
        code_graders=code_graders,
        model_graders=model_graders,
    )

    for trial in result.trials:
        path = save_eval_case(trial, task, model)
        print(f"  Recorded eval case: {path.name}")

    return result


# ── Load Functions ─────────────────────────────────────────────────


def load_eval_case(filepath: Path) -> dict:
    """Load a single eval case JSON file."""
    return json.loads(filepath.read_text())


def load_eval_cases_for_task(task_id: str) -> list[dict]:
    """Load all recorded eval cases for a given task ID, sorted by trial number."""
    if not RECORDED_CASES_DIR.exists():
        return []

    cases = []
    for filepath in sorted(RECORDED_CASES_DIR.glob(f"{task_id}_trial*.json")):
        cases.append(load_eval_case(filepath))

    cases.sort(key=lambda c: c["meta"]["trial_number"])
    return cases


def has_recorded_cases(task_id: str) -> bool:
    """Check if any recorded eval cases exist for a task."""
    if not RECORDED_CASES_DIR.exists():
        return False
    return any(RECORDED_CASES_DIR.glob(f"{task_id}_trial*.json"))


def reconstruct_trial(case: dict) -> HITLTrial:
    """Reconstruct an HITLTrial from a saved eval case.

    Graders consume the rebuilt trial identically to a live trial — they only
    read trial.transcript, trial.interrupts, trial.report, etc.
    """
    actual = case["actual"]
    meta = case["meta"]

    interrupts = [
        HITLInterruptRecord(
            interrupt_index=irpt["interrupt_index"],
            node_name=irpt["node_name"],
            hitl_type=irpt["hitl_type"],
            payload=irpt["payload"],
            resume_value=irpt["resume_value"],
            timestamp=irpt.get("timestamp", ""),
        )
        for irpt in actual["interrupts"]
    ]

    # Merge saved grades from both code and model sections
    saved_grades: dict[str, Any] = {}
    for name, val in case.get("grades", {}).get("code", {}).items():
        saved_grades[name] = val
    for name, val in case.get("grades", {}).get("model", {}).items():
        saved_grades[name] = val

    return HITLTrial(
        task_id=meta["task_id"],
        trial_number=meta["trial_number"],
        transcript=actual["transcript"],
        interrupts=interrupts,
        outcome=actual["outcome"],
        tool_calls_made=actual["tool_calls_made"],
        report=actual["report"],
        duration_seconds=actual["duration_seconds"],
        success=False,  # Re-computed by graders
        grades=saved_grades,
    )


# ── Offline Grade Function ────────────────────────────────────────


async def grade_recorded_cases(
    task: HITLEvalTask,
    k: int = 2,
    code_graders: dict | None = None,
    model_graders: dict | None = None,
) -> HITLEvalResult:
    """Phase 2 (Offline mode): load saved eval cases, re-grade, compute metrics.

    NO graph execution needed.
    """
    if code_graders is None:
        code_graders = {}
    if model_graders is None:
        model_graders = {}

    cases = load_eval_cases_for_task(task.id)
    if not cases:
        raise FileNotFoundError(
            f"No recorded eval cases for task '{task.id}'. "
            f"Run with --record flag first: pytest --record"
        )

    cases = cases[:k]

    trials = []
    for case in cases:
        trial = reconstruct_trial(case)

        # Clear previous grades and re-grade from scratch
        trial.grades = {}

        # Apply code graders
        all_code_pass = True
        for grader_name, grader_fn in code_graders.items():
            try:
                passed = grader_fn(trial)
            except Exception:
                passed = False
            trial.grades[grader_name] = passed
            if not passed:
                all_code_pass = False

        # Apply model graders
        all_model_pass = True
        for grader_name, grader_fn in model_graders.items():
            try:
                score = await grader_fn(trial)
            except Exception:
                score = 0.0
            trial.grades[grader_name] = score
            if score < 0.7:
                all_model_pass = False

        trial.success = all_code_pass and all_model_pass
        trials.append(trial)

    actual_k = len(trials)
    pk = pass_at_k(trials, actual_k)
    ppk = pass_pow_k(trials, actual_k)

    avg_grades: dict = {}
    if trials:
        all_grade_names: set[str] = set()
        for t in trials:
            all_grade_names.update(t.grades.keys())
        for name in all_grade_names:
            values = [t.grades[name] for t in trials if name in t.grades]
            if values:
                if isinstance(values[0], bool):
                    avg_grades[name] = sum(1 for v in values if v) / len(values)
                else:
                    avg_grades[name] = sum(values) / len(values)

    return HITLEvalResult(
        task_id=task.id,
        trials=trials,
        pass_at_k=pk,
        pass_pow_k=ppk,
        avg_grades=avg_grades,
    )


# ── Unified Run-or-Load Helper ────────────────────────────────────


async def run_or_load_eval(
    graph,
    task: HITLEvalTask,
    k: int = 2,
    code_graders: dict | None = None,
    model_graders: dict | None = None,
    record: bool = False,
    model: str = "gpt-4.1-mini",
) -> HITLEvalResult:
    """Unified entry point: record mode runs graph + saves; offline mode loads + grades.

    Args:
        graph: Compiled LangGraph (only used if record=True).
        task: HITLEvalTask definition.
        k: Number of trials.
        code_graders: Dict of name -> callable(HITLTrial) -> bool.
        model_graders: Dict of name -> async callable(HITLTrial) -> float.
        record: If True, run live graph and save eval cases first.
        model: Model identifier for metadata.
    """
    if record:
        # Phase 1: run real graph, save eval cases
        await record_eval_cases(
            graph=graph,
            task=task,
            k=k,
            code_graders=code_graders,
            model_graders=model_graders,
            model=model,
        )

    # Phase 2: load saved cases, re-grade
    return await grade_recorded_cases(
        task=task,
        k=k,
        code_graders=code_graders,
        model_graders=model_graders,
    )
