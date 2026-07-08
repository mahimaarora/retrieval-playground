#!/usr/bin/env python3
"""Smoke test for evaluation metrics (no live RAG / Qdrant required)."""

from retrieval_playground.src.evaluation.retrieval_metrics import RetrievalEvaluator
from retrieval_playground.src.evaluation.tool_metrics import ToolEvaluator, ToolTrace

QUESTION = "What is chain-of-thought prompting?"
REFERENCE_CONTEXT = (
    "A chain of thought is a series of intermediate natural language reasoning steps "
    "that lead to the final output, and we refer to this approach as chain-of-thought prompting."
)
REFERENCE_ANSWER = (
    "Chain-of-thought prompting provides exemplars with intermediate reasoning steps "
    "that lead to the final answer."
)
RETRIEVED = [
    REFERENCE_CONTEXT,
    "Chain-of-thought prompting improves reasoning on math benchmarks.",
    "Unrelated text about image classification.",
]
ANSWER = REFERENCE_ANSWER


def test_retrieval_metrics():
    evaluator = RetrievalEvaluator(k=3)
    result = evaluator.evaluate(
        questions=[QUESTION],
        contexts=[[RETRIEVED[0], RETRIEVED[1], RETRIEVED[2]]],
        reference_contexts=[REFERENCE_CONTEXT],
    )
    print("Retrieval (classical):", result.scores)
    assert result.scores["hit_rate_at_k@3"] > 0
    assert result.scores["keyword_overlap"] > 0


def test_tool_metrics():
    traces = [
        ToolTrace("q1", "vector_db", "vector_db", "hybrid_search", "hybrid_search"),
        ToolTrace("q2", "llm_direct", "llm_direct", None, None),
    ]
    result = ToolEvaluator().evaluate(traces)
    print("Tools:", result.scores)
    assert result.scores["tool_selection_accuracy"] == 1.0


if __name__ == "__main__":
    test_retrieval_metrics()
    test_tool_metrics()
    print("Smoke tests passed.")
