"""
Evaluation module using RAGAS for RAG system assessment.
"""

from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall
)
import numpy as np
from retrieval_playground.utils.pylogger import get_python_logger
from retrieval_playground.utils import constants
from retrieval_playground.utils.model_manager import model_manager
from ragas import RunConfig
from ragas.exceptions import RagasOutputParserException
from ragas.llms import LangchainLLMWrapper



class RAGEvaluator:
    """Evaluator for RAG systems using RAGAS metrics."""
    
    def __init__(self, metrics: List[str] = None):
        """
        Initialize the RAG evaluator.
        
        Args:
            metrics: List of metric names to compute. If None, computes all metrics.
                    Available: ['answer_relevancy', 'faithfulness', 'context_precision', 'context_recall']
        """
        self.logger = get_python_logger(log_level=constants.PYTHON_LOG_LEVEL)
        self.llm = model_manager.get_llm()
        
        # Define available metrics
        available_metrics = {
            'answer_relevancy': answer_relevancy,
            'faithfulness': faithfulness,
            'context_precision': context_precision,
            'context_recall': context_recall
        }
        
        # Set metrics to compute
        if metrics is None:
            self.metrics = list(available_metrics.values())
            self.metric_names = list(available_metrics.keys())
        else:
            self.metrics = [available_metrics[name] for name in metrics if name in available_metrics]
            self.metric_names = [name for name in metrics if name in available_metrics]
        
        self.logger.info(f"RAGEvaluator initialized with metrics: {self.metric_names}")
    
    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate a batch of QA pairs using RAGAS metrics.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context lists (each question has multiple context docs)
            ground_truths: List of reference/ground truth answers
            
        Returns:
            Dictionary containing metric scores
        """
        self.logger.info(f"🔄 Evaluating {len(questions)} QA pairs...")
        
        # Create evaluation dataset
        data = {
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts,
            "reference": ground_truths
        }
        
        dataset = Dataset.from_dict(data)
        
        # Run evaluation
        self.logger.info("🧮 Computing RAGAS metrics...")
        result = evaluate(dataset, metrics=self.metrics, llm=LangchainLLMWrapper(self.llm),
            embeddings=model_manager.get_embeddings(),
            run_config=RunConfig(
                timeout=180,
                max_retries=10,
                max_wait=60,
                max_workers=1,
                exception_types=RagasOutputParserException,
            ))
        
        # Extract scores for computed metrics only
        scores = {metric_name: result[metric_name] for metric_name in self.metric_names}
        
        self.logger.info("✅ Evaluation completed")
        return scores
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Evaluate a single QA pair.
        
        Args:
            question: The question
            answer: Generated answer
            contexts: List of context documents
            ground_truth: Reference answer
            
        Returns:
            Dictionary containing metric scores
        """
        return self.evaluate_batch(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth]
        )
    
    def evaluate_rag_results(
        self,
        rag_results: List[Dict[str, Any]],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate RAG results from BaselineRAG.query() output format.
        
        Args:
            rag_results: List of RAG query results from BaselineRAG
            ground_truths: List of reference answers
            
        Returns:
            Dictionary containing metric scores
        """
        questions = [result["question"] for result in rag_results]
        answers = [result["answer"] for result in rag_results]
        
        # Extract contexts from RAG results
        contexts = []
        for result in rag_results:
            context_texts = [ctx["content"] for ctx in result["context"]]
            contexts.append(context_texts)
        
        return self.evaluate_batch(questions, answers, contexts, ground_truths)


def evaluate_rag_system(
    rag_results: List[Dict[str, Any]],
    ground_truths: List[str],
    metrics: List[str] = None
) -> Dict[str, float]:
    """
    Convenience function to evaluate a RAG system.
    
    Args:
        rag_results: List of RAG query results
        ground_truths: List of reference answers
        metrics: List of metric names to compute. If None, computes all metrics.
        
    Returns:
        Dictionary containing metric scores
    """
    evaluator = RAGEvaluator(metrics=metrics)
    return evaluator.evaluate_rag_results(rag_results, ground_truths)
