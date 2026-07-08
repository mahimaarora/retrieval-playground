"""
Chunking Strategy Evaluation

Evaluates different chunking strategies to find the best one for the workshop dataset.
Focuses on text retrieval quality and multimodal content (tables/images) for Docling.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

from retrieval_playground.src.baseline_rag import RAG
from retrieval_playground.src.evaluation import RAGEvaluator
from retrieval_playground.src.pre_retrieval.chunking_manager import ChunkingStrategy
from retrieval_playground.utils.pylogger import get_python_logger
from retrieval_playground.utils import config


class ChunkingEvaluator:
    """Evaluates chunking strategies using RAGAS metrics and multimodal retrieval checks."""

    def __init__(self, test_queries_file: str = None, use_cloud: bool = True):
        """
        Initialize evaluator.

        Args:
            test_queries_file: Path to test queries JSON (defaults to workshop queries)
            use_cloud: Use cloud Qdrant (True) or local (False)
        """
        self.logger = get_python_logger(log_level=config.PYTHON_LOG_LEVEL)
        self.use_cloud = use_cloud

        # Load test queries
        if test_queries_file is None:
            test_queries_file = config.TEST_DATA_DIR / "chunking_test_queries.json"

        self.test_queries = self._load_queries(test_queries_file)
        self.logger.info(f"✅ Loaded {len(self.test_queries)} test queries")

        # RAGAS metrics
        self.metrics = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]
        self.rag_evaluator = RAGEvaluator(self.metrics)

    def _load_queries(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load test queries from JSON."""
        with open(file_path, 'r') as f:
            queries = json.load(f)

        # Log query types
        query_types = {}
        for q in queries:
            qtype = q.get('type', 'unknown')
            query_types[qtype] = query_types.get(qtype, 0) + 1

        self.logger.info(f"   Query types: {query_types}")
        return queries

    def _check_multimodal_content(self, result: Dict, query: Dict) -> Dict[str, bool]:
        """
        Check if tables/images were retrieved when needed.

        Only relevant for Docling - other strategies don't extract multimodal content.

        Returns:
            Dictionary with retrieval status for tables and images
        """
        check = {
            'needs_table': query.get('requires_table', False),
            'needs_image': query.get('requires_image', False),
            'found_table': False,
            'found_image': False
        }

        # Check retrieved chunks for multimodal content
        if 'contexts' in result:
            for context in result['contexts']:
                if isinstance(context, dict):
                    chunk_type = context.get('chunk_type', 'text')
                    if chunk_type == 'table':
                        check['found_table'] = True
                    elif chunk_type == 'image':
                        check['found_image'] = True

        return check

    def evaluate_strategy(self, strategy: ChunkingStrategy) -> Dict[str, Any]:
        """
        Evaluate a single chunking strategy.

        Steps:
        1. Initialize RAG with the strategy
        2. Run all test queries
        3. Calculate RAGAS metrics
        4. Check multimodal retrieval accuracy
        5. Return results

        Args:
            strategy: Which chunking strategy to evaluate

        Returns:
            Dictionary with all metrics
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"📊 Evaluating: {strategy.value.upper()}")
        self.logger.info(f"{'='*70}")

        # Initialize RAG
        rag = RAG(strategy=strategy, use_cloud=self.use_cloud)

        # Run queries
        self.logger.info(f"🔍 Running {len(self.test_queries)} queries...")
        rag_results = []
        multimodal_checks = []

        for idx, query_data in enumerate(self.test_queries, 1):
            question = query_data["user_input"]

            # Get answer
            result = rag.query(question, collection_name=strategy.value)
            rag_results.append(result)

            # Check multimodal content
            mm_check = self._check_multimodal_content(result, query_data)
            multimodal_checks.append(mm_check)

            if idx % 5 == 0:
                self.logger.info(f"   Processed {idx}/{len(self.test_queries)} queries")

        # Get RAGAS scores
        self.logger.info(f"📈 Computing RAGAS metrics...")
        ground_truths = [q["reference"] for q in self.test_queries]
        ragas_scores = self.rag_evaluator.evaluate_rag_results(rag_results, ground_truths)

        # Calculate multimodal accuracy
        multimodal_scores = self._calculate_multimodal_accuracy(multimodal_checks)

        # Combine results
        results = {
            'strategy': strategy.value,
            **ragas_scores,
            **multimodal_scores,
            'total_queries': len(self.test_queries)
        }

        self.logger.info(f"✅ {strategy.value} complete\n")

        # Cleanup
        rag.qdrant_client.close()
        del rag

        return results

    def _calculate_multimodal_accuracy(self, checks: List[Dict]) -> Dict[str, float]:
        """
        Calculate table and image retrieval accuracy.

        Returns:
            Accuracy scores for table and image retrieval
        """
        table_queries = [c for c in checks if c['needs_table']]
        image_queries = [c for c in checks if c['needs_image']]

        table_accuracy = 0.0
        if table_queries:
            table_success = sum(1 for c in table_queries if c['found_table'])
            table_accuracy = table_success / len(table_queries)

        image_accuracy = 0.0
        if image_queries:
            image_success = sum(1 for c in image_queries if c['found_image'])
            image_accuracy = image_success / len(image_queries)

        return {
            'table_accuracy': table_accuracy,
            'image_accuracy': image_accuracy,
            'multimodal_query_count': len(table_queries) + len(image_queries)
        }

    def evaluate_all_strategies(self) -> pd.DataFrame:
        """
        Evaluate all chunking strategies.

        Returns:
            DataFrame with results for all strategies
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("🚀 CHUNKING STRATEGY EVALUATION")
        self.logger.info("="*80 + "\n")

        # All strategies (fast to slow)
        strategies = [
            ChunkingStrategy.RECURSIVE_CHARACTER,
            ChunkingStrategy.PARENT_CHILD,
            ChunkingStrategy.CONTEXTUAL,
            ChunkingStrategy.DOCLING
        ]

        results = []

        for idx, strategy in enumerate(strategies, 1):
            self.logger.info(f"\n[{idx}/{len(strategies)}] {strategy.value}")

            try:
                result = self.evaluate_strategy(strategy)
                results.append(result)

                # Pause between strategies
                if idx < len(strategies):
                    self.logger.info("⏸️  Pausing 60s...\n")
                    time.sleep(60)

            except Exception as e:
                self.logger.error(f"❌ Failed: {e}")
                continue

        # Create DataFrame
        df = pd.DataFrame(results)

        # Calculate scores
        df['ragas_avg'] = df[self.metrics].mean(axis=1)
        df['multimodal_avg'] = df[['table_accuracy', 'image_accuracy']].mean(axis=1)

        # Combined: 70% RAGAS + 30% multimodal
        df['combined_score'] = (df['ragas_avg'] * 0.7) + (df['multimodal_avg'] * 0.3)

        self.logger.info("\n✅ Evaluation complete!\n")
        return df

    def print_results(self, df: pd.DataFrame):
        """Print evaluation results."""
        print("\n" + "="*90)
        print("📊 CHUNKING STRATEGY EVALUATION RESULTS")
        print("="*90)

        # Overall ranking
        df_sorted = df.sort_values('combined_score', ascending=False)

        print("\n🏆 OVERALL RANKING")
        print("-" * 90)
        for idx, (_, row) in enumerate(df_sorted.iterrows(), 1):
            emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "  "
            print(f"{emoji} {idx}. {row['strategy'].upper():<20} "
                  f"Combined: {row['combined_score']:.3f} "
                  f"(RAGAS: {row['ragas_avg']:.3f}, Multimodal: {row['multimodal_avg']:.3f})")

        # RAGAS breakdown
        print(f"\n📈 RAGAS METRICS")
        print("-" * 90)
        print(f"{'Strategy':<20} {'Answer Rel.':<12} {'Faithfulness':<12} {'Precision':<12} {'Recall':<12}")
        print("-" * 90)

        for _, row in df_sorted.iterrows():
            print(f"{row['strategy']:<20} "
                  f"{row['answer_relevancy']:<12.3f} "
                  f"{row['faithfulness']:<12.3f} "
                  f"{row['context_precision']:<12.3f} "
                  f"{row['context_recall']:<12.3f}")

        # Multimodal metrics
        print(f"\n🎨 MULTIMODAL RETRIEVAL (Tables & Images)")
        print("-" * 90)
        print(f"{'Strategy':<20} {'Table Accuracy':<18} {'Image Accuracy':<18} {'MM Queries':<15}")
        print("-" * 90)

        for _, row in df_sorted.iterrows():
            table_acc = f"{row['table_accuracy']:.1%}" if row['table_accuracy'] > 0 else "N/A"
            image_acc = f"{row['image_accuracy']:.1%}" if row['image_accuracy'] > 0 else "N/A"

            print(f"{row['strategy']:<20} {table_acc:<18} {image_acc:<18} {int(row['multimodal_query_count']):<15}")

        # Best per metric
        print(f"\n🌟 BEST BY METRIC")
        print("-" * 90)

        for metric in self.metrics + ['table_accuracy', 'image_accuracy']:
            best_strategy = df.loc[df[metric].idxmax(), 'strategy']
            best_score = df[metric].max()
            print(f"• {metric.replace('_', ' ').title():<30}: {best_strategy.upper():<20} ({best_score:.3f})")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 90)

        best = df_sorted.iloc[0]
        best_ragas = df.loc[df['ragas_avg'].idxmax()]
        best_mm = df.loc[df['multimodal_avg'].idxmax()]

        print(f"\n1. BEST OVERALL: {best['strategy'].upper()}")
        print(f"   Combined score: {best['combined_score']:.3f}")

        print(f"\n2. BEST FOR TEXT: {best_ragas['strategy'].upper()}")
        print(f"   RAGAS score: {best_ragas['ragas_avg']:.3f}")

        print(f"\n3. BEST FOR MULTIMODAL: {best_mm['strategy'].upper()}")
        print(f"   Multimodal score: {best_mm['multimodal_avg']:.3f}")

        # Docling insights
        docling_row = df[df['strategy'] == 'docling']
        if not docling_row.empty:
            docling = docling_row.iloc[0]
            print(f"\n4. DOCLING MULTIMODAL:")
            print(f"   Tables: {docling['table_accuracy']:.1%}")
            print(f"   Images: {docling['image_accuracy']:.1%}")

        print("\n" + "="*90 + "\n")

    def save_results(self, df: pd.DataFrame, output_file: str = None):
        """Save results to CSV."""
        if output_file is None:
            output_file = config.RESULTS_DIR / "chunking_evaluation_results.csv"

        df.to_csv(output_file, index=False)
        self.logger.info(f"💾 Saved to: {output_file}")


if __name__ == "__main__":
    """Run evaluation for all chunking strategies."""

    print("\n" + "="*90)
    print("🚀 CHUNKING STRATEGY EVALUATOR")
    print("="*90)
    print("\nEvaluating 4 strategies with 15 test queries")
    print("⏱️  Estimated time: ~10-15 minutes")
    print("="*90 + "\n")

    # Run evaluation
    evaluator = ChunkingEvaluator(use_cloud=True)
    df = evaluator.evaluate_all_strategies()
    evaluator.print_results(df)
    evaluator.save_results(df)

    print("✅ Done! Check CSV for detailed results.\n")
