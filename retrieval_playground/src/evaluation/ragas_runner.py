"""
Shared RAGAS runner used by generation (and optional retrieval) evaluation.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset

import retrieval_playground.utils.ragas_compat  # noqa: F401

from ragas import RunConfig, evaluate
from ragas.exceptions import RagasOutputParserException
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_recall,
    faithfulness,
)
from ragas.metrics import context_precision as default_context_precision

from retrieval_playground.utils import config
from retrieval_playground.utils.model_manager import model_manager
from retrieval_playground.utils.pylogger import get_python_logger

try:
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall

    RETRIEVAL_METRICS = {
        "context_precision": LLMContextPrecisionWithReference(),
        "context_recall": LLMContextRecall(),
    }
except ImportError:
    RETRIEVAL_METRICS = {
        "context_precision": default_context_precision,
        "context_recall": context_recall,
    }

GENERATION_METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
}

try:
    from ragas.metrics import AnswerAccuracy

    NVIDIA_METRICS = {
        "answer_accuracy": AnswerAccuracy(),
    }
except ImportError:
    NVIDIA_METRICS = {}

METRIC_REGISTRY = {**RETRIEVAL_METRICS, **GENERATION_METRICS, **NVIDIA_METRICS}


def _metric_result_key(metric: Any, requested_name: str) -> str:
    """RAGAS may return scores under the metric object's name, not our alias."""
    return getattr(metric, "name", requested_name)


def _event_loop_is_running() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _evaluate(dataset: Dataset, **kwargs: Any) -> Any:
    """
    Run ragas.evaluate safely from Jupyter notebooks.

    When IPython already has an event loop, repeated evaluate() calls can hit
    asyncio context errors on Python 3.12. Running in a worker thread with a
    fresh loop avoids that.
    """
    if not _event_loop_is_running():
        return evaluate(dataset, **kwargs)

    def _run_in_thread() -> Any:
        return evaluate(dataset, allow_nest_asyncio=False, **kwargs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_run_in_thread).result()


def _extract_scores(
    result: Any,
    metric_names: List[str],
    selected_metrics: Optional[List[Any]] = None,
) -> Dict[str, float]:
    key_pairs = [
        (name, _metric_result_key(metric, name))
        for name, metric in zip(
            metric_names,
            selected_metrics or metric_names,
        )
    ]

    scores: Dict[str, float] = {}
    try:
        df = result.to_pandas()
        for requested, actual in key_pairs:
            col = actual if actual in df.columns else requested
            if col in df.columns:
                scores[requested] = float(np.nanmean(df[col].tolist()))
        if len(scores) == len(metric_names):
            return scores
    except Exception:
        pass

    scores_dict = getattr(result, "_scores_dict", {})
    for requested, actual in key_pairs:
        key = actual if actual in scores_dict else requested
        values = scores_dict[key]
        if hasattr(values, "tolist"):
            values = values.tolist()
        scores[requested] = float(np.nanmean(values))
    return scores


def run_ragas(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    references: List[str],
    metrics: List[str],
    reference_contexts: Optional[List[List[str]]] = None,
) -> Dict[str, float]:
    """
    Run RAGAS metrics on a batch.

    `references` must be reference **answers** for LLM metrics.
    """
    logger = get_python_logger(log_level=config.PYTHON_LOG_LEVEL)
    metric_names = [name for name in metrics if name in METRIC_REGISTRY]
    if not metric_names:
        return {}

    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": references,
    }
    if reference_contexts is not None:
        data["reference_contexts"] = reference_contexts

    dataset = Dataset.from_dict(data)
    selected = [METRIC_REGISTRY[name] for name in metric_names]
    llm = model_manager.get_llm()

    logger.info(f"Running RAGAS metrics: {metric_names}")
    result = _evaluate(
        dataset,
        metrics=selected,
        llm=LangchainLLMWrapper(llm),
        embeddings=model_manager.get_embeddings(),
        run_config=RunConfig(
            timeout=180,
            max_retries=3,
            max_wait=60,
            max_workers=1,
            exception_types=RagasOutputParserException,
        ),
    )
    return _extract_scores(result, metric_names, selected)


class RAGEvaluator:
    """Backward-compatible wrapper around run_ragas()."""

    def __init__(self, metrics: Optional[List[str]] = None):
        self.metric_names = metrics or list(METRIC_REGISTRY.keys())

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> Dict[str, float]:
        return run_ragas(
            questions=questions,
            answers=answers,
            contexts=contexts,
            references=ground_truths,
            metrics=self.metric_names,
        )

    def evaluate_rag_results(
        self,
        rag_results: List[Dict[str, Any]],
        ground_truths: List[str],
    ) -> Dict[str, float]:
        return self.evaluate_batch(
            questions=[r["question"] for r in rag_results],
            answers=[r["answer"] for r in rag_results],
            contexts=[[c["content"] for c in r["context"]] for r in rag_results],
            ground_truths=ground_truths,
        )


def evaluate_rag_system(
    rag_results: List[Dict[str, Any]],
    ground_truths: List[str],
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    return RAGEvaluator(metrics=metrics).evaluate_rag_results(rag_results, ground_truths)
