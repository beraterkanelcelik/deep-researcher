"""Transcript and report persistence for HITL eval trials."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"


def _serialize_value(value: Any) -> Any:
    """Recursively serialize a value to JSON-safe types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return _serialize_value(value.model_dump())
    if hasattr(value, "__dict__"):
        return _serialize_value(vars(value))
    return str(value)


def _serialize_grades(grades: dict) -> dict:
    """Serialize a grades dict (bool or float values) to JSON-safe types."""
    result = {}
    for name, value in grades.items():
        if isinstance(value, bool):
            result[name] = value
        elif isinstance(value, (int, float)):
            result[name] = round(float(value), 4)
        else:
            result[name] = str(value)
    return result


def save_trial_transcript(trial, task) -> Path:
    """Save a full trial transcript to a JSON file.

    Args:
        trial: HITLTrial instance with transcript, interrupts, grades, etc.
        task: HITLEvalTask instance with id, grading_rubric, etc.

    Returns:
        Path to the saved JSON file.
    """
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{task.id}_trial{trial.trial_number}_{timestamp}.json"
    filepath = TRANSCRIPTS_DIR / filename

    data = {
        "task_id": task.id,
        "trial_number": trial.trial_number,
        "timestamp": timestamp,
        "duration_seconds": round(trial.duration_seconds, 2),
        "success": trial.success,
        "grades": _serialize_grades(trial.grades),
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
        "transcript": _serialize_value(trial.transcript),
        "tool_calls_made": trial.tool_calls_made,
        "report": _serialize_value(trial.report),
        "outcome": _serialize_value(trial.outcome),
        "grading_rubric": task.grading_rubric,
    }

    filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return filepath


def save_eval_report(result, task) -> Path:
    """Save an aggregate eval report to a JSON file.

    Args:
        result: HITLEvalResult instance with trials, pass_at_k, etc.
        task: HITLEvalTask instance.

    Returns:
        Path to the saved JSON file.
    """
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"report_{task.id}_{timestamp}.json"
    filepath = TRANSCRIPTS_DIR / filename

    trial_summaries = []
    for trial in result.trials:
        trial_summaries.append({
            "trial_number": trial.trial_number,
            "success": trial.success,
            "duration_seconds": round(trial.duration_seconds, 2),
            "interrupt_count": len(trial.interrupts),
            "grades": _serialize_grades(trial.grades),
        })

    data = {
        "task_id": result.task_id,
        "timestamp": timestamp,
        "pass_at_k": round(result.pass_at_k, 4),
        "pass_pow_k": round(result.pass_pow_k, 4),
        "avg_grades": _serialize_grades(result.avg_grades),
        "trials": trial_summaries,
        "grading_rubric": task.grading_rubric,
        "tags": task.tags,
    }

    filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return filepath
