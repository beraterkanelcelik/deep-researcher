"""Tests for Pydantic schemas in chat/schemas.py."""

import pytest
from pydantic import ValidationError

from chat.schemas import (
    ExplorerInstruction,
    HITLOption,
    HITLPayload,
    InstructionList,
    ResearchFinding,
    ResearchReport,
    SearchResult,
    SubTopicList,
    TaskItem,
    TaskPlan,
)

pytestmark = pytest.mark.unit


# ── TaskItem / TaskPlan ──────────────────────────────────────────────

class TestTaskItem:
    def test_task_item_valid(self):
        item = TaskItem(title="Setup DB", description="Configure PostgreSQL", priority="high")
        assert item.title == "Setup DB"
        assert item.description == "Configure PostgreSQL"
        assert item.priority == "high"

    def test_task_item_default_priority(self):
        item = TaskItem(title="Test", description="Run tests")
        assert item.priority == "medium"

    def test_task_item_invalid_priority(self):
        with pytest.raises(ValidationError):
            TaskItem(title="Bad", description="Bad priority", priority="critical")


class TestTaskPlan:
    def test_task_plan_valid(self):
        plan = TaskPlan(
            goal="Build an API",
            tasks=[
                TaskItem(title="Design", description="Design the API schema"),
                TaskItem(title="Implement", description="Write the code"),
            ],
            summary="A two-step plan for API development",
        )
        assert plan.goal == "Build an API"
        assert len(plan.tasks) == 2
        assert plan.summary.startswith("A two-step")


# ── SearchResult / ResearchFinding / ResearchReport ──────────────────

class TestSearchResult:
    def test_search_result_valid(self):
        sr = SearchResult(url="https://example.com", title="Example", content="Test content", score=0.95)
        assert sr.url == "https://example.com"
        assert sr.score == 0.95

    def test_search_result_defaults(self):
        sr = SearchResult()
        assert sr.url == ""
        assert sr.title == ""
        assert sr.content == ""
        assert sr.score == 0.0


class TestResearchFinding:
    def test_research_finding_valid(self):
        finding = ResearchFinding(
            insight="AI improves diagnostics",
            evidence="Studies show 95% accuracy",
            sources=["https://example.com/study"],
        )
        assert finding.insight == "AI improves diagnostics"
        assert len(finding.sources) == 1


class TestResearchReport:
    def test_research_report_complete(self):
        report = ResearchReport(
            title="AI in Healthcare",
            summary="A comprehensive review",
            key_findings=[
                ResearchFinding(insight="Finding 1", evidence="Evidence 1", sources=["url1"]),
            ],
            sources=["url1", "url2"],
            tags=["AI", "healthcare"],
            methodology="Web search and synthesis",
        )
        assert report.title == "AI in Healthcare"
        assert len(report.key_findings) == 1
        assert len(report.sources) == 2
        assert len(report.tags) == 2
        assert report.methodology == "Web search and synthesis"

    def test_research_report_minimal(self):
        report = ResearchReport(title="Minimal", summary="Just a summary")
        assert report.title == "Minimal"
        assert report.key_findings == []
        assert report.sources == []
        assert report.tags == []
        assert report.methodology == ""


# ── HITL Models ──────────────────────────────────────────────────────

class TestHITLOption:
    def test_hitl_option_valid(self):
        opt = HITLOption(id="opt-1", label="Option A", description="First option", selected=True)
        assert opt.id == "opt-1"
        assert opt.label == "Option A"
        assert opt.selected is True

    def test_hitl_option_defaults(self):
        opt = HITLOption(id="opt-2", label="Option B")
        assert opt.description == ""
        assert opt.selected is False


class TestHITLPayload:
    def test_hitl_payload_checkbox(self):
        payload = HITLPayload(
            hitl_type="checkbox",
            title="Select Topics",
            message="Choose topics to research",
            options=[
                HITLOption(id="t1", label="Topic 1"),
                HITLOption(id="t2", label="Topic 2"),
            ],
        )
        assert payload.hitl_type == "checkbox"
        assert len(payload.options) == 2
        assert payload.report is None

    @pytest.mark.parametrize("hitl_type", ["checkbox", "yes_no", "select", "text", "review", "confirm"])
    def test_hitl_payload_all_types(self, hitl_type):
        payload = HITLPayload(
            hitl_type=hitl_type,
            title=f"Test {hitl_type}",
            message=f"Message for {hitl_type}",
        )
        assert payload.hitl_type == hitl_type

    def test_hitl_payload_invalid_type(self):
        with pytest.raises(ValidationError):
            HITLPayload(hitl_type="invalid", title="Bad", message="Bad type")

    def test_hitl_payload_with_report(self):
        report = ResearchReport(title="Test Report", summary="Summary")
        payload = HITLPayload(
            hitl_type="review",
            title="Review",
            message="Review this",
            report=report,
        )
        assert payload.report is not None
        assert payload.report.title == "Test Report"


# ── Explorer Instruction Models ──────────────────────────────────────

class TestExplorerInstruction:
    def test_explorer_instruction_valid(self):
        inst = ExplorerInstruction(
            query="AI safety research 2025",
            search_focus="Recent papers and developments",
            context="Focus on alignment research",
        )
        assert inst.query == "AI safety research 2025"
        assert inst.search_focus == "Recent papers and developments"
        assert inst.context == "Focus on alignment research"

    def test_explorer_instruction_default_context(self):
        inst = ExplorerInstruction(query="test", search_focus="general")
        assert inst.context == ""


class TestInstructionList:
    def test_instruction_list_valid(self):
        ilist = InstructionList(
            instructions=[
                ExplorerInstruction(query="q1", search_focus="f1"),
                ExplorerInstruction(query="q2", search_focus="f2"),
            ]
        )
        assert len(ilist.instructions) == 2


class TestSubTopicList:
    def test_subtopic_list_valid(self):
        stl = SubTopicList(topics=["AI safety", "AI alignment", "AI governance"])
        assert len(stl.topics) == 3
        assert "AI safety" in stl.topics
