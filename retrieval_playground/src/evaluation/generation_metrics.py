"""
Generation-stage evaluation.

Uses RAGAS for faithfulness / answer relevancy / answer accuracy plus a simple length check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from retrieval_playground.src.evaluation.base import answer_length_ratio, mean_score
from retrieval_playground.src.evaluation.ragas_runner import run_ragas


@dataclass
class GenerationEvalResult:
    scores: Dict[str, float]
    ragas_scores: Dict[str, float]
    custom_scores: Dict[str, float]


class GenerationEvaluator:
    """Evaluate generated answers."""

    DEFAULT_RAGAS_METRICS = ["faithfulness", "answer_relevancy", "answer_accuracy"]

    def __init__(self, ragas_metrics: List[str] | None = None):
        self.ragas_metrics = ragas_metrics or self.DEFAULT_RAGAS_METRICS

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> GenerationEvalResult:
        ragas_scores = run_ragas(
            questions=questions,
            answers=answers,
            contexts=contexts,
            references=ground_truths,
            metrics=self.ragas_metrics,
        )

        custom_scores = {
            "answer_length_ratio": mean_score(
                [
                    answer_length_ratio(answer, truth)
                    for answer, truth in zip(answers, ground_truths)
                ]
            )
        }

        return GenerationEvalResult(
            scores={**ragas_scores, **custom_scores},
            ragas_scores=ragas_scores,
            custom_scores=custom_scores,
        )

    def evaluate_rag_results(
        self,
        rag_results: List[Dict],
        ground_truths: List[str],
    ) -> GenerationEvalResult:
        return self.evaluate(
            questions=[r["question"] for r in rag_results],
            answers=[r["answer"] for r in rag_results],
            contexts=[[c["content"] for c in r["context"]] for r in rag_results],
            ground_truths=ground_truths,
        )
