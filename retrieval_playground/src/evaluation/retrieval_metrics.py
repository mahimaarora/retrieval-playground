"""
Retrieval-stage evaluation.

Uses deterministic metrics that compare retrieved chunks against gold evidence.
RAGAS LLM retrieval metrics are available via `evaluate_ragas()` using reference answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from retrieval_playground.src.evaluation.base import (
    MetricResult,
    keyword_overlap,
    mean_score,
    tokenize,
)
from retrieval_playground.src.evaluation.ragas_runner import run_ragas


def _is_relevant(chunk: str, reference: str, threshold: float = 0.15) -> bool:
    ref_tokens = tokenize(reference)
    if not ref_tokens:
        return False
    return len(ref_tokens & tokenize(chunk)) / len(ref_tokens) >= threshold


def hit_rate_at_k(
    contexts: List[List[str]],
    references: List[str],
    k: int = 3,
    threshold: float = 0.15,
) -> float:
    hits = []
    for chunks, reference in zip(contexts, references):
        top_k = chunks[:k]
        hits.append(
            float(any(_is_relevant(chunk, reference, threshold) for chunk in top_k))
        )
    return mean_score(hits)


def mean_reciprocal_rank(
    contexts: List[List[str]],
    references: List[str],
    threshold: float = 0.15,
) -> float:
    scores = []
    for chunks, reference in zip(contexts, references):
        rank = None
        for index, chunk in enumerate(chunks, start=1):
            if _is_relevant(chunk, reference, threshold):
                rank = index
                break
        scores.append(1.0 / rank if rank else 0.0)
    return mean_score(scores)


def keyword_overlap_batch(
    contexts: List[List[str]],
    references: List[str],
) -> float:
    scores = [keyword_overlap(chunks, reference) for chunks, reference in zip(contexts, references)]
    return mean_score(scores)


@dataclass
class RetrievalEvalResult:
    scores: Dict[str, float]


class RetrievalEvaluator:
    """Evaluate whether retrieval surfaced the right evidence."""

    def __init__(self, k: int = 3):
        self.k = k

    def evaluate(
        self,
        questions: List[str],
        contexts: List[List[str]],
        reference_contexts: List[str],
    ) -> RetrievalEvalResult:
        return RetrievalEvalResult(
            scores={
                f"hit_rate_at_k@{self.k}": hit_rate_at_k(
                    contexts, reference_contexts, k=self.k
                ),
                "mrr": mean_reciprocal_rank(contexts, reference_contexts),
                "keyword_overlap": keyword_overlap_batch(contexts, reference_contexts),
            }
        )

    def evaluate_ragas(
        self,
        questions: List[str],
        contexts: List[List[str]],
        reference_answers: List[str],
        answers: List[str],
    ) -> Dict[str, float]:
        """
        RAGAS retrieval metrics judged against reference **answers**.

        Requires generated answers because the LLM judge compares chunks to the
        reference answer claims, not the gold evidence passage.
        """
        return run_ragas(
            questions=questions,
            answers=answers,
            contexts=contexts,
            references=reference_answers,
            metrics=["context_precision", "context_recall"],
        )
