"""
Adaptive Retrieval: Automatic configuration based on query complexity

Automatically adjusts:
- k (number of results)
- retrieval method (dense vs hybrid vs multi-query)
- reranking (on/off)

Simple queries → Fast (k=3, dense, no reranking)
Moderate queries → Balanced (k=5, hybrid, reranking)
Complex queries → Comprehensive (k=8, multi-query, reranking)
"""

from typing import List, Dict
from langchain_core.documents import Document

from retrieval_playground.utils.query_classifier import classify_query_complexity
from retrieval_playground.src.mid_retrieval.hybrid_search import HybridRetriever
from retrieval_playground.src.mid_retrieval.reranking import Reranker
from retrieval_playground.src.mid_retrieval.multi_query_hybrid import MultiQueryHybrid


class AdaptiveRetriever:
    """
    Automatically adapts retrieval configuration to query complexity.

    Complexity levels:
    - simple → k=3, dense search, no reranking
    - moderate → k=5, hybrid search, reranking
    - complex → k=8, multi-query (3 variants), reranking

    Example:
        retriever = AdaptiveRetriever(collection_name="hybrid")
        results = retriever.search("What is BERT?")
    """

    def __init__(
        self,
        collection_name: str = "hybrid",
        reranker_collection: str = "recursive_character",
        use_cloud: bool = True
    ):
        """
        Args:
            collection_name: Qdrant collection for hybrid search
            reranker_collection: Collection for reranker
            use_cloud: Use cloud Qdrant vs local
        """
        self.collection_name = collection_name

        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(collection_name=collection_name)

        # Initialize reranker
        self.reranker = Reranker(
            collection_name=reranker_collection,
            top_k=20,
            top_n=8,  # Will be overridden by config
            use_cloud=use_cloud
        )

        # Initialize multi-query hybrid (for complex queries)
        self.multi_query_hybrid = MultiQueryHybrid(
            collection_name=collection_name,
            reranker_collection=reranker_collection,
            num_variants=3,
            use_cloud=use_cloud
        )

        print("✅ Adaptive retriever initialized")

    def _get_config(self, complexity: dict) -> dict:
        """Get configuration based on complexity level."""
        level = complexity["complexity"]

        configs = {
            "simple": {
                "method": "dense",
                "k": 3,
                "use_reranking": False
            },
            "moderate": {
                "method": "hybrid",
                "k": 5,
                "use_reranking": True
            },
            "complex": {
                "method": "multi_query",
                "k": 8,
                "use_reranking": True,
                "num_variants": 3
            }
        }

        return configs.get(level, configs["moderate"])

    def search(self, query: str) -> List[Document]:
        """
        Adaptive search with automatic configuration.

        Args:
            query: Search query

        Returns:
            Retrieved documents (with reranking if configured)
        """
        # Step 1: Analyze complexity
        complexity = classify_query_complexity(query)

        # Step 2: Get configuration
        config = self._get_config(complexity)

        # Step 3: Execute retrieval based on method
        if config["method"] == "dense":
            # Dense only (fast for simple queries)
            docs = self.hybrid_retriever._dense_search(query, k=config["k"])

        elif config["method"] == "hybrid":
            # Hybrid (BM25 + Dense for moderate queries)
            docs = self.hybrid_retriever.search(query, k=config["k"])

            # Apply reranking if configured
            if config["use_reranking"] and docs:
                docs = self.reranker._rerank(docs, query)
                docs = docs[:config["k"]]

        elif config["method"] == "multi_query":
            # Multi-query hybrid (for complex queries)
            docs = self.multi_query_hybrid.search(query, k=config["k"])

        else:
            docs = []

        return docs

    def compare_configs(self, query: str) -> Dict:
        """
        Compare results using all complexity configurations.

        Args:
            query: Search query

        Returns:
            Dictionary with results from each config level
        """
        comparison = {}

        for level in ["simple", "moderate", "complex"]:
            mock_complexity = {"complexity": level, "score": 0}
            config = self._get_config(mock_complexity)

            # Execute with this config
            if config["method"] == "dense":
                docs = self.hybrid_retriever._dense_search(query, k=config["k"])

            elif config["method"] == "hybrid":
                docs = self.hybrid_retriever.search(query, k=config["k"])
                if config["use_reranking"] and docs:
                    docs = self.reranker._rerank(docs, query)
                    docs = docs[:config["k"]]

            elif config["method"] == "multi_query":
                docs = self.multi_query_hybrid.search(query, k=config["k"])

            else:
                docs = []

            comparison[level] = {
                "config": config,
                "results": docs
            }

        return comparison

    def close(self):
        """Close Qdrant connections."""
        self.hybrid_retriever.close()
        self.reranker.close()
        self.multi_query_hybrid.close()


if __name__ == "__main__":
    """
    Run: python -m retrieval_playground.src.mid_retrieval.adaptive_retrieval
    """

    retriever = AdaptiveRetriever(
        collection_name="hybrid",
        reranker_collection="recursive_character",
        use_cloud=True
    )

    # Test queries (different complexity levels)
    queries = [
        ("What is Agent Laboratory?", "simple"),
        ("Compare PyTorch and JAX for scientific computing", "moderate"),
        ("How does reinforcement learning work in the context of scientific research and what effect does it have on experimental design?", "complex"),
    ]

    print("\n" + "=" * 80)
    print("ADAPTIVE RETRIEVAL - Complexity-Based Configuration")
    print("=" * 80)

    for query, expected_complexity in queries:
        # Analyze complexity
        complexity = classify_query_complexity(query)
        config = retriever._get_config(complexity)

        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print(f"Complexity: {complexity['complexity']} (score: {complexity['score']}/5)")
        print(f"Expected: {expected_complexity}")
        print(f"Config: method={config['method']}, k={config['k']}, reranking={config['use_reranking']}")
        print(f"{'=' * 80}")

        # Execute search
        results = retriever.search(query)
        print(f"\nRetrieved: {len(results)} documents\n")

        # Show top 2 results
        for i, doc in enumerate(results[:2], 1):
            if config["use_reranking"]:
                score = doc.metadata.get('rerank_score', 0)
                print(f"{i}. Rerank Score: {score:.4f}")
            print(f"   {doc.page_content[:120]}...\n")

    print("=" * 80 + "\n")
    retriever.close()
