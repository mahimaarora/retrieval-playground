"""
End-to-end evaluation pipeline across retrieval, generation, and tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from retrieval_playground.src.evaluation.generation_metrics import GenerationEvaluator
from retrieval_playground.src.evaluation.retrieval_metrics import RetrievalEvaluator
from retrieval_playground.src.evaluation.tool_metrics import ToolEvaluator, ToolTrace


@dataclass
class PipelineEvalResult:
    retrieval: Dict[str, float] = field(default_factory=dict)
    generation: Dict[str, float] = field(default_factory=dict)
    tools: Dict[str, float] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for stage, scores in [
            ("retrieval", self.retrieval),
            ("generation", self.generation),
            ("tools", self.tools),
        ]:
            for metric, value in scores.items():
                rows.append({"stage": stage, "metric": metric, "score": value})
        return pd.DataFrame(rows)


class RAGEvaluationPipeline:
    def __init__(
        self,
        ragas_metrics: List[str] | None = None,
        retrieval_k: int = 3
    ):
        self.retrieval_evaluator = RetrievalEvaluator(k=retrieval_k)
        self.generation_evaluator = GenerationEvaluator(ragas_metrics=ragas_metrics)
        self.tool_evaluator = ToolEvaluator()

    def evaluate_all(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
        reference_contexts: List[str],
        tool_traces: Optional[List[ToolTrace]] = None,
        include_ragas_retrieval: bool = True,
    ) -> PipelineEvalResult:
        retrieval_scores = self.retrieval_evaluator.evaluate(
            questions, contexts, reference_contexts
        ).scores

        if include_ragas_retrieval:
            retrieval_scores.update(
                self.retrieval_evaluator.evaluate_ragas(
                    questions, contexts, ground_truths, answers
                )
            )

        generation_scores = self.generation_evaluator.evaluate(
            questions, answers, contexts, ground_truths
        ).scores

        tool_scores = (
            self.tool_evaluator.evaluate(tool_traces).scores if tool_traces else {}
        )

        return PipelineEvalResult(
            retrieval=retrieval_scores,
            generation=generation_scores,
            tools=tool_scores,
        )

    def evaluate_rag_results(
        self,
        rag_results: List[Dict[str, Any]],
        ground_truths: List[str],
        reference_contexts: List[str],
        tool_traces: Optional[List[ToolTrace]] = None,
    ) -> PipelineEvalResult:
        return self.evaluate_all(
            questions=[r["question"] for r in rag_results],
            answers=[r["answer"] for r in rag_results],
            contexts=[[c["content"] for c in r["context"]] for r in rag_results],
            ground_truths=ground_truths,
            reference_contexts=reference_contexts,
            tool_traces=tool_traces,
        )
