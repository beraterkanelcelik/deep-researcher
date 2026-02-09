"""Integration tests for the deep research subgraph."""

import pytest

from chat.research_graph import (
    build_research_subgraph,
    explorer_node,
    orchestrate_node,
    synthesize_node,
)
from chat.schemas import InstructionList

pytestmark = [pytest.mark.integration]


class TestResearchSubgraphBuilds:
    def test_research_subgraph_builds(self):
        graph = build_research_subgraph()
        compiled = graph.compile()
        assert compiled is not None

    def test_research_subgraph_nodes(self):
        graph = build_research_subgraph()
        compiled = graph.compile()
        node_names = set(compiled.get_graph().nodes.keys())
        expected = {"clarify", "orchestrate", "explorer", "synthesize", "review"}
        assert expected.issubset(node_names)


class TestOrchestrateNode:
    def test_orchestrate_generates_instructions(self):
        state = {
            "clarified_topics": ["AI safety alignment", "AI evaluation methods"],
            "depth": "standard",
        }
        result = orchestrate_node(state)
        assert "explorer_instructions" in result
        assert len(result["explorer_instructions"]) > 0
        for inst in result["explorer_instructions"]:
            assert "query" in inst
            assert "search_focus" in inst
        assert result["status"] == "orchestrated"


class TestExplorerNode:
    def test_explorer_returns_search_results(self):
        state = {
            "query": "AI safety research 2025",
            "search_focus": "Recent developments",
            "context": "",
            "search_results": [],
        }
        result = explorer_node(state)
        assert "search_results" in result
        assert len(result["search_results"]) > 0
        first = result["search_results"][0]
        assert "content" in first
        assert "title" in first


class TestSynthesizeNode:
    def test_synthesize_produces_report(self):
        state = {
            "topic": "AI Safety",
            "clarified_topics": ["alignment", "evaluation"],
            "search_results": [
                {
                    "title": "AI Safety Research",
                    "url": "https://example.com/ai-safety",
                    "content": "AI safety focuses on ensuring AI systems behave as intended.",
                    "score": 0.9,
                },
                {
                    "title": "RLHF Methods",
                    "url": "https://example.com/rlhf",
                    "content": "Reinforcement learning from human feedback is a key technique.",
                    "score": 0.85,
                },
            ],
        }
        result = synthesize_node(state)
        assert "report" in result
        report = result["report"]
        assert "title" in report
        assert "summary" in report
        assert "key_findings" in report
        assert isinstance(report["key_findings"], list)
        assert result["status"] == "synthesized"
