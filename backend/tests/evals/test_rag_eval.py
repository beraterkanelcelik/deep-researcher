"""Eval: RAG retrieval relevance and answer grounding.

Uploads sample documents, runs queries through the graph,
then verifies answers contain expected content.
"""

import io

import pytest

from .graders import grade_content_contains, grade_response_quality
from .harness import EvalTask, Trial, run_eval_task

pytestmark = [pytest.mark.eval, pytest.mark.slow, pytest.mark.django_db]


@pytest.fixture
def ingest_rag_documents(rag_accuracy_tasks):
    """Ingest all sample documents from the RAG dataset."""
    from chat.rag.ingest import ingest_document

    for entry in rag_accuracy_tasks:
        file = io.BytesIO(entry["document"].encode("utf-8"))
        ingest_document(file, f"rag_eval_{entry['id']}.txt")


async def test_rag_retrieval_accuracy(eval_graph, rag_accuracy_tasks, ingest_rag_documents):
    """Verify that RAG-augmented answers contain expected terms."""
    results = []

    for entry in rag_accuracy_tasks:
        task = EvalTask(
            id=entry["id"],
            input_messages=[{"role": "human", "content": entry["query"]}],
            expected_content=entry["expected_in_answer"],
            tags=["rag"],
        )

        def code_grader(trial, expected=entry["expected_in_answer"]):
            return grade_content_contains(trial, expected)

        result = await run_eval_task(
            graph=eval_graph,
            task=task,
            k=1,
            code_grader=code_grader,
        )
        results.append(result)

    print(f"\n{'='*60}")
    print("RAG Retrieval Accuracy Eval Results")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r.pass_at_k >= 1.0 else "FAIL"
        print(f"  {r.task_id}: pass@1={r.pass_at_k:.2f} [{status}]")

    total = sum(r.pass_at_k for r in results) / len(results)
    print(f"{'='*60}")
    print(f"  Aggregate accuracy: {total:.2f}")
    print(f"{'='*60}")

    assert total >= 0.5, f"RAG accuracy too low: {total:.2f}"


async def test_rag_answer_quality(eval_graph, rag_accuracy_tasks, ingest_rag_documents):
    """Verify that RAG answers are grounded in documents (model grader)."""
    entry = rag_accuracy_tasks[0]

    task = EvalTask(
        id=entry["id"],
        input_messages=[{"role": "human", "content": entry["query"]}],
        grading_rubric="Answer is grounded in the provided document context, accurate, and concise",
    )

    async def model_grader(trial):
        return await grade_response_quality(
            trial.transcript,
            "Answer is grounded in the provided document context, accurate, and concise",
        )

    result = await run_eval_task(
        graph=eval_graph,
        task=task,
        k=1,
        model_grader=model_grader,
    )

    print(f"\nRAG answer quality for {entry['id']}: {result.avg_model_score:.2f}")
    assert result.avg_model_score >= 0.5
