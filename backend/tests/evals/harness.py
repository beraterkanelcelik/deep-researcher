"""Eval harness: core abstractions for running agent evaluations.

Based on Anthropic's "Demystifying Evals for AI Agents" article.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class EvalTask:
    """A single evaluation task definition."""

    id: str
    input_messages: list[dict]
    expected_tools: list[str] | None = None
    expected_content: list[str] | None = None
    grading_rubric: str | None = None
    max_turns: int = 10
    tags: list[str] = field(default_factory=list)


@dataclass
class Trial:
    """Result of a single trial run of an eval task."""

    task_id: str
    trial_number: int
    transcript: list[dict]
    outcome: dict
    tool_calls_made: list[str]
    duration_seconds: float
    success: bool
    model_score: float | None = None


@dataclass
class EvalResult:
    """Aggregate result for an eval task across multiple trials."""

    task_id: str
    trials: list[Trial]
    pass_at_k: float
    pass_pow_k: float
    avg_model_score: float | None = None


async def run_eval_task(
    graph,
    task: EvalTask,
    k: int = 3,
    code_grader=None,
    model_grader=None,
) -> EvalResult:
    """Run an eval task k times and compute metrics.

    Args:
        graph: Compiled LangGraph to invoke.
        task: The eval task definition.
        k: Number of trials.
        code_grader: Callable(Trial) -> bool for code-based grading.
        model_grader: Async callable(Trial) -> float for model-based grading.

    Returns:
        EvalResult with trials and computed metrics.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    from .metrics import pass_at_k, pass_pow_k

    trials = []

    for trial_num in range(k):
        start = time.time()

        lc_messages = []
        for msg in task.input_messages:
            role = msg.get("type", msg.get("role", "human"))
            content = msg.get("content", "")
            if role == "human":
                lc_messages.append(HumanMessage(content=content))
            elif role == "ai":
                lc_messages.append(AIMessage(content=content))

        result = await graph.ainvoke(
            {
                "messages": lc_messages,
                "model": "gpt-4.1-mini",
                "research_reports": [],
                "current_plan": None,
                "pending_save": None,
                "research_request": None,
                "topic": "",
                "depth": "",
                "report": None,
            }
        )

        duration = time.time() - start

        # Extract transcript
        transcript = []
        tool_calls_made = []
        for m in result["messages"]:
            entry = {"type": getattr(m, "type", "unknown"), "content": getattr(m, "content", "")}
            if hasattr(m, "tool_calls") and m.tool_calls:
                entry["tool_calls"] = m.tool_calls
                for tc in m.tool_calls:
                    tool_calls_made.append(tc["name"])
            if hasattr(m, "name") and m.name:
                entry["name"] = m.name
            transcript.append(entry)

        # Grade
        trial = Trial(
            task_id=task.id,
            trial_number=trial_num,
            transcript=transcript,
            outcome={"messages_count": len(result["messages"])},
            tool_calls_made=tool_calls_made,
            duration_seconds=duration,
            success=False,
        )

        if code_grader:
            trial.success = code_grader(trial)

        if model_grader:
            trial.model_score = await model_grader(trial)
            # If no code grader, use model score threshold
            if code_grader is None:
                trial.success = trial.model_score >= 0.7

        trials.append(trial)

    # Compute metrics
    pk = pass_at_k(trials, k)
    ppk = pass_pow_k(trials, k)
    avg_score = None
    scores = [t.model_score for t in trials if t.model_score is not None]
    if scores:
        avg_score = sum(scores) / len(scores)

    return EvalResult(
        task_id=task.id,
        trials=trials,
        pass_at_k=pk,
        pass_pow_k=ppk,
        avg_model_score=avg_score,
    )


# ── HITL Eval Data Structures ────────────────────────────────────────


@dataclass
class HITLResponse:
    """A simulated user response for an HITL interrupt point."""

    expected_hitl_type: str
    resume_value: Any
    description: str = ""


@dataclass
class HITLInterruptRecord:
    """Record of a single HITL interrupt encountered during a trial."""

    interrupt_index: int
    node_name: str
    hitl_type: str
    payload: dict
    resume_value: Any
    timestamp: str = ""


@dataclass
class HITLEvalTask:
    """Eval task definition that includes simulated HITL responses."""

    id: str
    input_messages: list[dict]
    hitl_responses: list[HITLResponse]
    expected_hitl_types: list[str]
    grading_rubric: str | None = None
    max_interrupts: int = 10
    tags: list[str] = field(default_factory=list)


@dataclass
class HITLTrial:
    """Result of a single HITL eval trial with interrupt records and per-grader grades."""

    task_id: str
    trial_number: int
    transcript: list[dict]
    interrupts: list[HITLInterruptRecord]
    outcome: dict
    tool_calls_made: list[str]
    report: dict | None
    duration_seconds: float
    success: bool
    grades: dict = field(default_factory=dict)


@dataclass
class HITLEvalResult:
    """Aggregate result for an HITL eval task across multiple trials."""

    task_id: str
    trials: list[HITLTrial]
    pass_at_k: float
    pass_pow_k: float
    avg_grades: dict = field(default_factory=dict)


def _get_fallback_response(hitl_type: str) -> Any:
    """Return a safe default resume value when pre-defined responses are exhausted."""
    fallbacks = {
        "checkbox": [],
        "yes_no": True,
        "select": None,
        "text": "",
        "review": {"action": "approve"},
        "confirm": {"action": "save"},
    }
    return fallbacks.get(hitl_type, {})


def _extract_transcript(messages) -> tuple[list[dict], list[str]]:
    """Extract transcript entries and tool call names from LangChain messages."""
    transcript = []
    tool_calls_made = []
    for m in messages:
        entry = {
            "type": getattr(m, "type", "unknown"),
            "content": getattr(m, "content", ""),
        }
        if hasattr(m, "tool_calls") and m.tool_calls:
            entry["tool_calls"] = m.tool_calls
            for tc in m.tool_calls:
                tool_calls_made.append(tc["name"])
        if hasattr(m, "name") and m.name:
            entry["name"] = m.name
        transcript.append(entry)
    return transcript, tool_calls_made


async def run_hitl_eval_task(
    graph,
    task: HITLEvalTask,
    k: int = 2,
    code_graders: dict | None = None,
    model_graders: dict | None = None,
) -> HITLEvalResult:
    """Run an HITL eval task k times with interrupt/resume cycles.

    Args:
        graph: Compiled LangGraph with a checkpointer (e.g. MemorySaver).
        task: HITL eval task with simulated responses.
        k: Number of trials.
        code_graders: Dict of name -> callable(HITLTrial) -> bool.
        model_graders: Dict of name -> async callable(HITLTrial) -> float.

    Returns:
        HITLEvalResult with trials, pass@k, pass^k, and per-grader averages.
    """
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.types import Command

    from .metrics import pass_at_k as calc_pass_at_k
    from .metrics import pass_pow_k as calc_pass_pow_k

    if code_graders is None:
        code_graders = {}
    if model_graders is None:
        model_graders = {}

    trials = []

    for trial_num in range(k):
        start = time.time()
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Build initial messages
        lc_messages = []
        for msg in task.input_messages:
            role = msg.get("type", msg.get("role", "human"))
            content = msg.get("content", "")
            if role == "human":
                lc_messages.append(HumanMessage(content=content))
            elif role == "ai":
                lc_messages.append(AIMessage(content=content))

        initial_input = {
            "messages": lc_messages,
            "model": "gpt-4.1-mini",
            "research_reports": [],
            "current_plan": None,
            "pending_save": None,
            "research_request": None,
            "topic": "",
            "depth": "",
            "report": None,
        }

        # First invocation — runs until first interrupt or completion
        try:
            result = await graph.ainvoke(initial_input, config)
        except Exception:
            result = None

        # Interrupt/resume loop
        interrupt_records: list[HITLInterruptRecord] = []
        response_idx = 0

        for loop_iter in range(task.max_interrupts):
            state = await graph.aget_state(config)

            # Check for interrupts in state.tasks
            if not state.tasks:
                break

            # Find the first interrupt
            interrupt_found = False
            for state_task in state.tasks:
                if hasattr(state_task, "interrupts") and state_task.interrupts:
                    for irpt in state_task.interrupts:
                        payload = irpt.value if hasattr(irpt, "value") else irpt
                        if isinstance(payload, (list, tuple)):
                            payload = payload[0] if payload else {}
                        if not isinstance(payload, dict):
                            payload = {}

                        hitl_type = payload.get("hitl_type", "unknown")
                        node_name = state_task.name if hasattr(state_task, "name") else "unknown"

                        # Pick resume value from task responses or fallback
                        if response_idx < len(task.hitl_responses):
                            resume_value = task.hitl_responses[response_idx].resume_value
                        else:
                            resume_value = _get_fallback_response(hitl_type)

                        interrupt_records.append(
                            HITLInterruptRecord(
                                interrupt_index=len(interrupt_records),
                                node_name=node_name,
                                hitl_type=hitl_type,
                                payload=payload,
                                resume_value=resume_value,
                                timestamp=datetime.now(timezone.utc).isoformat(),
                            )
                        )

                        response_idx += 1
                        interrupt_found = True
                        break  # Handle one interrupt at a time
                if interrupt_found:
                    break

            if not interrupt_found:
                break

            # Resume the graph with the response value
            try:
                result = await graph.ainvoke(
                    Command(resume=interrupt_records[-1].resume_value), config
                )
            except Exception:
                result = None
                break

        duration = time.time() - start

        # Extract transcript from final state
        transcript = []
        tool_calls_made = []
        report = None

        if result and "messages" in result:
            transcript, tool_calls_made = _extract_transcript(result["messages"])
            report = result.get("report")
            if not report and result.get("research_reports"):
                report = result["research_reports"][-1]
        else:
            # Try to get messages from state
            final_state = await graph.aget_state(config)
            if final_state.values and "messages" in final_state.values:
                transcript, tool_calls_made = _extract_transcript(
                    final_state.values["messages"]
                )
                report = final_state.values.get("report")
                if not report and final_state.values.get("research_reports"):
                    report = final_state.values["research_reports"][-1]

        trial = HITLTrial(
            task_id=task.id,
            trial_number=trial_num,
            transcript=transcript,
            interrupts=interrupt_records,
            outcome={
                "messages_count": len(transcript),
                "interrupt_count": len(interrupt_records),
            },
            tool_calls_made=tool_calls_made,
            report=report,
            duration_seconds=duration,
            success=False,
            grades={},
        )

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

    # Compute aggregate metrics
    pk = calc_pass_at_k(trials, k)
    ppk = calc_pass_pow_k(trials, k)

    # Average grades across trials
    avg_grades: dict = {}
    if trials:
        all_grade_names = set()
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
