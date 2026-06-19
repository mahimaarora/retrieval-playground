"""
Adaptive Retrieval: Smart configuration based on query complexity

Automatically configures:
- k (number of results)
- threshold (minimum score)
- retrieval method (dense, hybrid, multi-query)
- reranker model (flashrank, bge, cohere)

Simple queries → Fast, efficient
Complex queries → Comprehensive, high-quality

Simple to use:
    adaptive = AdaptiveRetriever()
    results = adaptive.search("your query here")
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from retrieval_playground.utils import config, constants
from retrieval_playground.utils.model_manager import model_manager
from retrieval_playground.src.pre_retrieval.chunking_manager import ChunkingStrategy
from retrieval_playground.src.pre_retrieval.query_rephrasing import classify_query_complexity
from retrieval_playground.src.mid_retrieval.hybrid_search import HybridRetriever

# Optional reranking import
try:
    from retrieval_playground.src.mid_retrieval.reranking import Reranker
    RERANKING_AVAILABLE = True
except ImportError:
    RERANKING_AVAILABLE = False


class AdaptiveRetriever:
    """
    Smart retriever that adapts to query complexity.

    How it works:
    1. Analyzes query complexity (simple/moderate/complex)
    2. Configures retrieval parameters automatically
    3. Selects appropriate reranker model
    4. Returns optimized results

    Example:
        adaptive = AdaptiveRetriever()

        # Simple query → k=2, dense search, fast reranker
        results = adaptive.search("What is BERT?")

        # Complex query → k=8, hybrid search, powerful reranker
        results = adaptive.search("Compare BERT and GPT-3 architectures...")
    """

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
        use_cloud: bool = True,
        qdrant_client: Optional[QdrantClient] = None
    ):
        """
        Initialize adaptive retriever.

        Args:
            strategy: Chunking strategy to use
            use_cloud: Use cloud Qdrant (True) or local (False)
            qdrant_client: Optional pre-configured Qdrant client
        """
        self.strategy = strategy

        # Setup Qdrant client
        if use_cloud:
            self.qdrant_client = QdrantClient(
                url=constants.QDRANT_URL,
                api_key=constants.QDRANT_KEY
            )
        elif qdrant_client is None:
            qdrant_path = config.QDRANT_DIR / strategy.value
            self.qdrant_client = QdrantClient(path=str(qdrant_path))
        else:
            self.qdrant_client = qdrant_client

        # Setup vector store
        self.embeddings = model_manager.get_embeddings()
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=strategy.value,
            embedding=self.embeddings
        )

        # Initialize hybrid retriever (lazy loading)
        self._hybrid_retriever = None

        print("✅ Adaptive retriever initialized")

    def _get_hybrid_retriever(self) -> HybridRetriever:
        """Get or create hybrid retriever (lazy loading)."""
        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetriever(
                strategy=self.strategy,
                use_cloud=False,  # Reuse existing client
                qdrant_client=self.qdrant_client
            )
        return self._hybrid_retriever

    def _get_config(self, complexity: dict) -> dict:
        """
        Get retrieval configuration based on complexity.

        Args:
            complexity: Complexity analysis from classify_query_complexity()

        Returns:
            Configuration dictionary
        """
        level = complexity["complexity"]

        configs = {
            "simple": {
                "method": "dense",
                "k": 2,
                "threshold": 0.7,
                "reranker": "flashrank",
                "use_reranking": False  # Fast queries don't need reranking
            },
            "moderate": {
                "method": "hybrid",
                "k": 5,
                "threshold": 0.5,
                "reranker": "bge",
                "use_reranking": True
            },
            "complex": {
                "method": "hybrid",
                "k": 8,
                "threshold": 0.3,
                "reranker": "bge",
                "use_reranking": True
            }
        }

        return configs.get(level, configs["moderate"])

    def search(
        self,
        query: str,
        force_config: Optional[dict] = None,
        verbose: bool = False
    ) -> List[Document]:
        """
        Adaptive search with automatic configuration.

        Args:
            query: Search query
            force_config: Optional manual configuration override
            verbose: Print configuration details

        Returns:
            Retrieved and optionally reranked documents

        Example:
            # Automatic configuration
            results = adaptive.search("What is BERT?")

            # Manual override
            results = adaptive.search(
                "What is BERT?",
                force_config={"method": "hybrid", "k": 10}
            )

            # With details
            results = adaptive.search("What is BERT?", verbose=True)
        """
        # Step 1: Analyze complexity
        complexity = classify_query_complexity(query)

        # Step 2: Get configuration
        if force_config:
            config = force_config
        else:
            config = self._get_config(complexity)

        if verbose:
            print(f"\n🔍 Query: {query[:80]}...")
            print(f"\n📊 Complexity Analysis:")
            print(f"   Level: {complexity['complexity']}")
            print(f"   Score: {complexity['score']}/5")
            print(f"   Signals: {complexity['signals']}")
            print(f"\n⚙️  Configuration:")
            print(f"   Method: {config['method']}")
            print(f"   k: {config['k']}")
            print(f"   Threshold: {config['threshold']}")
            print(f"   Reranker: {config['reranker']}")
            print(f"   Use Reranking: {config['use_reranking']}\n")

        # Step 3: Execute retrieval based on method
        if config["method"] == "dense":
            # Dense search only (fast for simple queries)
            results = self.vector_store.similarity_search_with_score(
                query,
                k=config["k"]
            )
            # Convert to Document list
            docs = [doc for doc, score in results if score >= config["threshold"]]

        elif config["method"] == "hybrid":
            # Hybrid search (BM25 + Dense)
            hybrid = self._get_hybrid_retriever()
            results = hybrid.search(query, k=config["k"])
            # Filter by threshold (using RRF score)
            docs = [
                doc for doc in results
                if doc.metadata.get('rrf_score', 0) >= config["threshold"] * 0.1
            ]

        else:
            # Default to dense
            results = self.vector_store.similarity_search(query, k=config["k"])
            docs = results

        # Step 4: Apply reranking if configured
        if config.get("use_reranking", False) and docs and RERANKING_AVAILABLE:
            try:
                reranker = Reranker(
                    strategy=self.strategy,
                    use_cloud=False,
                    qdrant_client=self.qdrant_client,
                    model=config["reranker"],
                    top_n=config["k"]
                )

                # Rerank the documents
                if hasattr(reranker, '_rerank_bge') or hasattr(reranker, '_rerank_flashrank'):
                    # Use direct reranking method
                    if config["reranker"] == "bge":
                        docs = reranker._rerank_bge(docs, query)
                    elif config["reranker"] == "flashrank":
                        docs = reranker._rerank_flashrank(docs, query)
                    else:
                        # Default reranker
                        docs = reranker.retrieve(query)
                else:
                    docs = reranker.retrieve(query)

                if verbose:
                    print(f"✅ Reranked with {config['reranker']}")

            except Exception as e:
                if verbose:
                    print(f"⚠️  Reranking skipped: {e}")
                # Continue without reranking
        elif config.get("use_reranking", False) and not RERANKING_AVAILABLE:
            if verbose:
                print("⚠️  Reranking not available (import error)")

        return docs

    def compare_configs(self, query: str) -> dict:
        """
        Compare results from different configurations.

        Useful for understanding how complexity affects retrieval!

        Args:
            query: Search query

        Returns:
            Dictionary with results from each complexity level

        Example:
            comparison = adaptive.compare_configs("What is BERT?")
            print("Simple:", len(comparison['simple']))
            print("Moderate:", len(comparison['moderate']))
            print("Complex:", len(comparison['complex']))
        """
        complexity_levels = ["simple", "moderate", "complex"]
        comparison = {}

        for level in complexity_levels:
            # Create mock complexity
            mock_complexity = {"complexity": level, "score": 0}
            config = self._get_config(mock_complexity)

            # Get results
            results = self.search(query, force_config=config)
            comparison[level] = results

        return comparison

    def close(self):
        """Close Qdrant client connection."""
        self.qdrant_client.close()
        if self._hybrid_retriever:
            self._hybrid_retriever.close()


# Simple helper function
def adaptive_search(
    query: str,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
    verbose: bool = False
) -> List[Document]:
    """
    Quick adaptive search - no class initialization needed.

    Args:
        query: Search query
        strategy: Chunking strategy
        verbose: Print configuration details

    Returns:
        Retrieved documents

    Example:
        from mid_retrieval.adaptive_retrieval import adaptive_search

        results = adaptive_search("What is BERT?", verbose=True)
    """
    retriever = AdaptiveRetriever(strategy=strategy)
    results = retriever.search(query, verbose=verbose)
    retriever.close()
    return results


# Example usage
if __name__ == "__main__":
    # Initialize
    adaptive = AdaptiveRetriever(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    # Test queries with different complexity
    queries = [
        "What is BERT?",  # Simple
        "How does attention mechanism work?",  # Moderate
        "Compare BERT and GPT-3 architectures and explain their differences"  # Complex
    ]

    for query in queries:
        print("="*60)
        results = adaptive.search(query, verbose=True)
        print(f"Retrieved {len(results)} documents\n")

    adaptive.close()
