"""
Evaluation script for comparing different chunking strategies.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

from retrieval_playground.src.baseline_rag import RAG
from retrieval_playground.src.evaluation import RAGEvaluator
from retrieval_playground.src.pre_retrieval.chunking_strategies import ChunkingStrategy
from retrieval_playground.utils.pylogger import get_python_logger
from retrieval_playground.utils import constants, config


class ChunkingEvaluator:
    """Evaluator for different chunking strategies."""
    
    def __init__(self, query_count, metrics):
        """Initialize the chunking evaluator."""
        self.logger = get_python_logger(log_level=constants.PYTHON_LOG_LEVEL)
        self.rag_evaluator = RAGEvaluator(metrics)
        self.metrics = metrics

        # Load test queries
        self.test_queries = self._load_test_queries()[:query_count]
        self.logger.info(f"Loaded {len(self.test_queries)} test queries")

    def _load_test_queries(self) -> List[Dict[str, Any]]:
        """Load test queries from JSON file."""
        queries_path = config.TESTS_DIR / "test_queries.json"
        with open(queries_path, 'r') as f:
            return json.load(f)

    def evaluate_strategy(self, strategy: ChunkingStrategy) -> Dict[str, float]:
        """Evaluate a single chunking strategy."""
        self.logger.info(f"Evaluating {strategy.value} strategy...")

        # Initialize RAG with specific collection path
        rag = RAG(strategy=strategy)

        # Process queries
        print("Generating answers for queries...")
        rag_results = []
        for query_data in self.test_queries:
            question = query_data["user_input"]
            result = rag.query(question, collection_name=strategy.value)
            rag_results.append(result)

        # Extract ground truths
        ground_truths = [q["reference"] for q in self.test_queries]

        # Evaluate using RAGAS
        scores = self.rag_evaluator.evaluate_rag_results(rag_results, ground_truths)

        self.logger.info(f"✅ {strategy.value} evaluation completed")
        rag.qdrant_client.close()
        del rag
        return scores

    def evaluate_all_strategies(self) -> pd.DataFrame:
        """Evaluate all chunking strategies and return comparison results."""
        self.logger.info("Starting evaluation of all chunking strategies...")

        strategies = [
            ChunkingStrategy.BASELINE,
            ChunkingStrategy.RECURSIVE_CHARACTER,
            ChunkingStrategy.UNSTRUCTURED,
            ChunkingStrategy.DOCLING
        ]

        results = []
        for strategy in strategies:
            try:
                scores = self.evaluate_strategy(strategy)
                result = {"strategy": strategy.value, "queries": [q["user_input"] for q in self.test_queries], **scores}
                result = pd.DataFrame({k: v if isinstance(v, list) else [v]*len(result['faithfulness'])
                   for k, v in result.items()})
                results.append(result)
                self.logger.info("Pausing for a minute ...")
                time.sleep(60)
            except Exception as e:
                self.logger.error(f"❌ Failed to evaluate {strategy.value}: {e}")
                continue

        # Create comparison DataFrame
        df = pd.concat(results).reset_index(drop=True)

        # Sort by overall performance (average of all metrics)
        df["average_score"] = df[self.metrics].mean(axis=1)
        df = df.sort_values(by=["queries","average_score"], ascending=False)

        self.logger.info("✅ All strategies evaluated successfully")
        return df

    def print_results(self, df: pd.DataFrame):
        """Print evaluation results in a clear, readable format."""
        print("\n" + "="*80)
        print("📊 CHUNKING STRATEGY EVALUATION RESULTS")
        print("="*80)

        # Get unique strategies and their average performance
        strategy_summary = df.groupby('strategy').agg({
            'answer_relevancy': 'mean',
            'faithfulness': 'mean',
            'context_precision': 'mean',
            'context_recall': 'mean',
            'average_score': 'mean'
        }).round(3)

        # Sort by average score
        strategy_summary = strategy_summary.sort_values('average_score', ascending=False)

        print(f"\nSTRATEGY RANKINGS (by average score):")
        print("-" * 50)
        for i, (strategy, row) in enumerate(strategy_summary.iterrows(), 1):
            print(f"{i}. {strategy.upper():<20} | Avg: {row['average_score']:.3f}")

        print(f"\nDETAILED METRICS:")
        print("-" * 80)
        print(f"{'Strategy':<20} {'Relevancy':<10} {'Faithful':<10} {'Precision':<10} {'Recall':<10} {'Average':<10}")
        print("-" * 80)

        for strategy, row in strategy_summary.iterrows():
            print(f"{strategy:<20} {row['answer_relevancy']:<10.3f} {row['faithfulness']:<10.3f} "
                  f"{row['context_precision']:<10.3f} {row['context_recall']:<10.3f} {row['average_score']:<10.3f}")

        # Show best and worst performing strategies
        best_strategy = strategy_summary.index[0]
        worst_strategy = strategy_summary.index[-1]
        improvement = strategy_summary.loc[best_strategy, 'average_score'] - strategy_summary.loc[worst_strategy, 'average_score']

        print(f"\nKEY INSIGHTS:")
        print("-" * 30)
        print(f"• Best Strategy: {best_strategy.upper()} ({strategy_summary.loc[best_strategy, 'average_score']:.3f})")
        print(f"• Worst Strategy: {worst_strategy.upper()} ({strategy_summary.loc[worst_strategy, 'average_score']:.3f})")
        print(f"• Performance Gap: {improvement:.3f} ({improvement/strategy_summary.loc[worst_strategy, 'average_score']*100:.1f}% improvement)")

        # Show which strategy excels in each metric
        print(f"\nBEST BY METRIC:")
        print("-" * 25)
        for metric in self.metrics:
            best_for_metric = strategy_summary[metric].idxmax()
            score = strategy_summary.loc[best_for_metric, metric]
            print(f"• {metric.replace('_', ' ').title():<18}: {best_for_metric.upper()} ({score:.3f})")

        print("="*80 + "\n")


if __name__ == "__main__":
    evaluator = ChunkingEvaluator()
    df = evaluator.evaluate_all_strategies()
    evaluator.print_results(df)
