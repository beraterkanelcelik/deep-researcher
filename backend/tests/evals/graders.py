"""Code-based and model-based graders for agent evaluations."""

import json
import os

from .harness import HITLTrial, Trial

# ── Code-Based Graders (deterministic) ───────────────────────────────


def grade_tool_selection(trial: Trial, expected_tools: list[str]) -> bool:
    """Check if agent called the expected tools."""
    if not expected_tools:
        # If no tools expected, none should have been called
        return len(trial.tool_calls_made) == 0
    return all(t in trial.tool_calls_made for t in expected_tools)


def grade_content_contains(trial: Trial, patterns: list[str]) -> bool:
    """Check if final AI response contains expected patterns (case-insensitive)."""
    # Find the last AI message in transcript
    final_content = ""
    for entry in reversed(trial.transcript):
        if entry.get("type") == "ai" and entry.get("content"):
            final_content = entry["content"]
            break

    return all(p.lower() in final_content.lower() for p in patterns)


def grade_no_hallucinated_tools(trial: Trial) -> bool:
    """Check agent didn't call tools that don't exist."""
    valid = {
        "get_current_time",
        "calculator",
        "tavily_search",
        "create_plan",
        "deep_research",
        "save_report",
    }
    return all(t in valid for t in trial.tool_calls_made)


def grade_hitl_payload_structure(payload: dict, expected_type: str) -> bool:
    """Validate HITL payload has correct structure for its type."""
    required = {"hitl_type", "title", "message"}
    if not required.issubset(payload.keys()):
        return False
    if payload["hitl_type"] != expected_type:
        return False
    if expected_type in ("checkbox", "select") and "options" not in payload:
        return False
    if expected_type in ("review", "confirm") and "report" not in payload:
        return False
    return True


def grade_report_structure(report: dict) -> bool:
    """Check that a research report has the required structure."""
    required_keys = {"title", "summary", "key_findings", "sources"}
    if not required_keys.issubset(report.keys()):
        return False
    if not report["title"]:
        return False
    if not report["summary"]:
        return False
    if not isinstance(report["key_findings"], list):
        return False
    if not isinstance(report["sources"], list):
        return False
    return True


# ── Model-Based Graders (LLM-as-judge with gpt-4.1-mini) ────────────


async def _call_judge(prompt: str) -> float:
    """Call gpt-4.1-mini as a judge and extract a 0-1 score."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        temperature=0,
    )

    response = await llm.ainvoke(prompt)
    content = response.content.strip()

    # Extract numeric score from response
    # Look for a number between 0 and 1
    import re

    matches = re.findall(r"\b(0\.\d+|1\.0|0|1)\b", content)
    if matches:
        return float(matches[-1])

    # Fallback: look for X/10 or X/100 patterns
    matches = re.findall(r"(\d+)\s*/\s*10", content)
    if matches:
        return float(matches[-1]) / 10

    return 0.5  # Default if parsing fails


async def grade_response_quality(transcript: list[dict], rubric: str) -> float:
    """Score 0-1 on response quality using LLM judge."""
    conversation = ""
    for entry in transcript:
        role = entry.get("type", "unknown")
        content = entry.get("content", "")
        if content:
            conversation += f"{role}: {content}\n\n"

    prompt = (
        "You are an eval grader. Score this AI assistant conversation on a scale of 0.0 to 1.0.\n\n"
        f"Grading rubric: {rubric}\n\n"
        f"Conversation:\n{conversation}\n\n"
        "Respond with ONLY a decimal number between 0.0 and 1.0. "
        "1.0 means perfect, 0.0 means completely fails the rubric."
    )
    return await _call_judge(prompt)


async def grade_research_groundedness(report: dict, sources: list[dict]) -> float:
    """Score 0-1: are claims in the report supported by sources?"""
    report_text = json.dumps(report, indent=2)
    sources_text = json.dumps(sources[:10], indent=2) if sources else "No sources provided"

    prompt = (
        "You are an eval grader evaluating research report groundedness.\n\n"
        "Score 0.0 to 1.0: Are the claims and findings in this report supported by the sources?\n\n"
        f"Report:\n{report_text}\n\n"
        f"Sources:\n{sources_text}\n\n"
        "Respond with ONLY a decimal number between 0.0 and 1.0."
    )
    return await _call_judge(prompt)


async def grade_research_coverage(report: dict, topic: str) -> float:
    """Score 0-1: does the report cover key aspects of the topic?"""
    report_text = json.dumps(report, indent=2)

    prompt = (
        "You are an eval grader evaluating research report coverage.\n\n"
        f"Score 0.0 to 1.0: Does this report comprehensively cover the topic '{topic}'?\n\n"
        f"Report:\n{report_text}\n\n"
        "Consider: breadth of subtopics, depth of analysis, practical insights.\n"
        "Respond with ONLY a decimal number between 0.0 and 1.0."
    )
    return await _call_judge(prompt)


async def grade_conversational_tone(transcript: list[dict]) -> float:
    """Score 0-1 on helpfulness, clarity, and appropriate tone."""
    conversation = ""
    for entry in transcript:
        role = entry.get("type", "unknown")
        content = entry.get("content", "")
        if content:
            conversation += f"{role}: {content}\n\n"

    prompt = (
        "You are an eval grader evaluating conversational quality.\n\n"
        "Score 0.0 to 1.0 on: helpfulness, clarity, appropriate tone, and engagement.\n\n"
        f"Conversation:\n{conversation}\n\n"
        "Respond with ONLY a decimal number between 0.0 and 1.0."
    )
    return await _call_judge(prompt)


# ── HITL-Specific Graders ─────────────────────────────────────────────


def grade_hitl_flow_completeness(
    trial: HITLTrial, expected_hitl_types: list[str]
) -> bool:
    """Check that the trial encountered the expected HITL interrupt sequence.

    Uses subsequence matching: expected types must appear in order within
    actual types. This allows the agent to trigger additional interrupts
    (e.g. confirm after review) without failing.
    """
    actual_types = [rec.hitl_type for rec in trial.interrupts]
    it = iter(actual_types)
    return all(expected in it for expected in expected_hitl_types)


def grade_hitl_payload_all_valid(trial: HITLTrial) -> bool:
    """Validate that every interrupt payload in the trial has correct structure."""
    if not trial.interrupts:
        return False
    for rec in trial.interrupts:
        if not grade_hitl_payload_structure(rec.payload, rec.hitl_type):
            return False
    return True


def grade_research_report_present(trial: HITLTrial) -> bool:
    """Check that the trial produced a valid research report."""
    if trial.report is None:
        return False
    return grade_report_structure(trial.report)


async def grade_deep_research_quality(trial: HITLTrial) -> float:
    """LLM-judge score (0-1) on deep research quality.

    Evaluates: report completeness, topic coverage, finding quality,
    source quality, and coherence. Includes interrupt flow summary
    in the judge prompt for context.
    """
    if trial.report is None:
        return 0.0

    report_text = json.dumps(trial.report, indent=2, default=str)

    # Build interrupt flow summary
    interrupt_summary = "HITL Interrupt Flow:\n"
    for rec in trial.interrupts:
        interrupt_summary += (
            f"  - Interrupt #{rec.interrupt_index}: "
            f"type={rec.hitl_type}, node={rec.node_name}\n"
        )

    prompt = (
        "You are an eval grader evaluating a deep research report produced by an AI agent.\n\n"
        "Score 0.0 to 1.0 based on these criteria:\n"
        "1. Report completeness: Does it have a title, summary, key findings, and sources?\n"
        "2. Topic coverage: Are the findings relevant and covering the topic breadth?\n"
        "3. Finding quality: Are the key findings specific, insightful, and supported by evidence?\n"
        "4. Source quality: Are sources provided and relevant?\n"
        "5. Coherence: Is the report well-structured and logically organized?\n\n"
        f"{interrupt_summary}\n"
        f"Research Report:\n{report_text}\n\n"
        "Respond with ONLY a decimal number between 0.0 and 1.0. "
        "1.0 means excellent research report, 0.0 means completely inadequate."
    )
    return await _call_judge(prompt)
