"""Eval: HITL payload correctness at all interrupt points."""

import operator
import uuid
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from chat.nodes import save_confirm_node
from chat.research_graph import clarify_node, review_node
from chat.schemas import HITLPayload
from .graders import grade_hitl_payload_structure

pytestmark = [pytest.mark.eval]


def _extract_interrupt_payload(compiled_graph, config):
    """Extract the first interrupt payload from graph state."""
    state = compiled_graph.get_state(config)
    for task in state.tasks:
        if hasattr(task, "interrupts") and task.interrupts:
            for irpt in task.interrupts:
                payload = irpt.value if hasattr(irpt, "value") else irpt
                if isinstance(payload, (list, tuple)):
                    payload = payload[0] if payload else {}
                return payload
    return None


class TestHITLPayloadStructure:
    def test_clarify_checkpoint_payload(self):
        """clarify_node should produce a valid checkbox HITL payload."""
        from langgraph.checkpoint.memory import MemorySaver

        from chat.research_graph import DeepResearchState, ResearchInput, ResearchOutput

        graph = StateGraph(DeepResearchState, input=ResearchInput, output=ResearchOutput)
        graph.add_node("clarify", clarify_node)
        graph.add_edge(START, "clarify")
        graph.add_edge("clarify", END)
        compiled = graph.compile(checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        compiled.invoke({"topic": "Machine Learning", "depth": "standard"}, config)

        payload = _extract_interrupt_payload(compiled, config)
        assert payload is not None, "Expected interrupt from clarify_node"
        assert isinstance(payload, dict)
        assert grade_hitl_payload_structure(payload, "checkbox")
        assert len(payload.get("options", [])) > 0

    def test_review_checkpoint_payload(self):
        """review_node should produce a valid review HITL payload."""
        from langgraph.checkpoint.memory import MemorySaver

        class ReviewTestState(TypedDict):
            report: dict | None
            status: str

        graph = StateGraph(ReviewTestState)
        graph.add_node("review", review_node)
        graph.add_edge(START, "review")
        graph.add_edge("review", END)
        compiled = graph.compile(checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        compiled.invoke(
            {
                "report": {
                    "title": "Test Report",
                    "summary": "Summary",
                    "key_findings": [],
                    "sources": [],
                    "tags": [],
                    "methodology": "test",
                },
                "status": "",
            },
            config,
        )

        payload = _extract_interrupt_payload(compiled, config)
        assert payload is not None, "Expected interrupt from review_node"
        assert isinstance(payload, dict)
        assert grade_hitl_payload_structure(payload, "review")
        assert payload.get("report") is not None

    def test_save_confirm_checkpoint_payload(self):
        """save_confirm_node should produce a valid confirm HITL payload."""
        from langgraph.checkpoint.memory import MemorySaver

        class SaveConfirmTestState(MessagesState):
            research_reports: list
            pending_save: dict | None

        graph = StateGraph(SaveConfirmTestState)
        graph.add_node("save_confirm", save_confirm_node)
        graph.add_edge(START, "save_confirm")
        graph.add_edge("save_confirm", END)
        compiled = graph.compile(checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        compiled.invoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "save_report", "id": "tc-1", "args": {"report_index": 0}}
                        ],
                    )
                ],
                "research_reports": [
                    {
                        "title": "Report",
                        "summary": "Summary",
                        "key_findings": [],
                        "sources": [],
                        "tags": [],
                        "methodology": "",
                    }
                ],
                "pending_save": None,
            },
            config,
        )

        payload = _extract_interrupt_payload(compiled, config)
        assert payload is not None, "Expected interrupt from save_confirm_node"
        assert isinstance(payload, dict)
        assert grade_hitl_payload_structure(payload, "confirm")
        assert payload.get("report") is not None


class TestHITLPayloadPydantic:
    @pytest.mark.parametrize(
        "hitl_type,extra",
        [
            ("checkbox", {"options": [{"id": "1", "label": "opt1", "description": "", "selected": False}]}),
            ("yes_no", {}),
            ("select", {"options": [{"id": "1", "label": "opt1", "description": "", "selected": False}]}),
            ("text", {}),
            ("review", {"report": {"title": "T", "summary": "S", "key_findings": [], "sources": [], "tags": [], "methodology": ""}}),
            ("confirm", {"report": {"title": "T", "summary": "S", "key_findings": [], "sources": [], "tags": [], "methodology": ""}}),
        ],
    )
    def test_hitl_payload_validates(self, hitl_type, extra):
        payload = HITLPayload(
            hitl_type=hitl_type,
            title=f"Test {hitl_type}",
            message="Test message",
            **extra,
        )
        payload_dict = payload.model_dump()
        assert grade_hitl_payload_structure(payload_dict, hitl_type)
