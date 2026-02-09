"""Tests for tool functions in chat/tools.py."""

import json

import pytest

from chat.tools import (
    ALL_TOOLS,
    SIMPLE_TOOLS,
    calculator,
    create_plan,
    deep_research,
    get_current_time,
    save_report,
    tavily_search,
)

pytestmark = pytest.mark.unit


# ── calculator ───────────────────────────────────────────────────────

class TestCalculator:
    def test_calculator_addition(self):
        result = calculator.invoke({"expression": "2 + 3"})
        assert "5" in result

    def test_calculator_complex(self):
        result = calculator.invoke({"expression": "(10 * 5) + 3"})
        assert "53" in result

    def test_calculator_division(self):
        result = calculator.invoke({"expression": "10 / 3"})
        assert "3.333" in result

    def test_calculator_security_blocks_import(self):
        result = calculator.invoke({"expression": "__import__('os')"})
        assert "error" in result.lower() or "invalid" in result.lower()

    def test_calculator_security_blocks_exec(self):
        result = calculator.invoke({"expression": "exec('print(1)')"})
        assert "error" in result.lower() or "invalid" in result.lower()

    def test_calculator_invalid_expression(self):
        result = calculator.invoke({"expression": "foo bar"})
        assert "error" in result.lower() or "invalid" in result.lower()

    def test_calculator_power(self):
        result = calculator.invoke({"expression": "2 ** 10"})
        assert "1024" in result


# ── get_current_time ─────────────────────────────────────────────────

class TestGetCurrentTime:
    def test_get_current_time(self):
        import datetime

        result = get_current_time.invoke({})
        # Should be in ISO-like format: YYYY-MM-DD HH:MM:SS
        assert len(result) == 19
        parsed = datetime.datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        now = datetime.datetime.now()
        diff = abs((now - parsed).total_seconds())
        assert diff < 2


# ── tavily_search ────────────────────────────────────────────────────

class TestTavilySearch:
    @pytest.mark.integration
    def test_tavily_search(self):
        result = tavily_search.invoke({"query": "Python programming"})
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) > 0
        first = parsed[0]
        assert "url" in first
        assert "content" in first

    @pytest.mark.integration
    def test_tavily_search_returns_results(self):
        result = tavily_search.invoke({"query": "latest AI news"})
        parsed = json.loads(result)
        assert len(parsed) >= 1


# ── create_plan ──────────────────────────────────────────────────────

class TestCreatePlan:
    @pytest.mark.integration
    def test_create_plan(self):
        result = create_plan.invoke({"goal": "Build a REST API"})
        parsed = json.loads(result)
        assert "tasks" in parsed
        assert isinstance(parsed["tasks"], list)
        for task in parsed["tasks"]:
            assert "title" in task
            assert "description" in task

    @pytest.mark.integration
    def test_create_plan_has_steps(self):
        result = create_plan.invoke({"goal": "Design a database"})
        parsed = json.loads(result)
        assert len(parsed["tasks"]) >= 2


# ── Sentinel tools ───────────────────────────────────────────────────

class TestSentinelTools:
    def test_deep_research_sentinel(self):
        result = deep_research.invoke({"topic": "quantum computing"})
        assert isinstance(result, str)
        assert "routing" in result.lower() or "subgraph" in result.lower()

    def test_save_report_sentinel(self):
        result = save_report.invoke({"report_index": 0})
        assert isinstance(result, str)
        assert "routing" in result.lower() or "save" in result.lower()


# ── Tool exports ─────────────────────────────────────────────────────

class TestToolExports:
    def test_simple_tools_count(self):
        assert len(SIMPLE_TOOLS) == 4

    def test_all_tools_count(self):
        assert len(ALL_TOOLS) == 6

    def test_simple_tools_names(self):
        names = {t.name for t in SIMPLE_TOOLS}
        assert names == {"get_current_time", "calculator", "tavily_search", "create_plan"}

    def test_all_tools_includes_sentinels(self):
        names = {t.name for t in ALL_TOOLS}
        assert "deep_research" in names
        assert "save_report" in names
