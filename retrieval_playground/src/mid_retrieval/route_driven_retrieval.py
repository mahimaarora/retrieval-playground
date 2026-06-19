"""
Route-Driven Retrieval: Use semantic routes to control retrieval

Automatically:
- Detects query type (factual, comparison, analytical, etc.)
- Selects appropriate retrieval method
- Chooses right tool (vector_db, sql, web)
- Decides when to use reranking

No manual configuration needed!

Simple to use:
    route_retriever = RouteRetriever()
    results = route_retriever.search("your query")
"""

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from retrieval_playground.utils import config, constants
from retrieval_playground.utils.model_manager import model_manager
from retrieval_playground.src.pre_retrieval.chunking_manager import ChunkingStrategy
from retrieval_playground.src.pre_retrieval.routing import route_with_complexity_analysis
from retrieval_playground.src.mid_retrieval.hybrid_search import HybridRetriever

# Optional imports
try:
    from retrieval_playground.src.pre_retrieval.query_rephrasing import expand_query, reciprocal_rank_fusion
    MULTI_QUERY_AVAILABLE = True
except ImportError:
    MULTI_QUERY_AVAILABLE = False


class RouteRetriever:
    """
    Smart retriever that uses semantic routing.

    How it works:
    1. Routes query to appropriate category (factual, comparison, etc.)
    2. Each route has pre-configured retrieval settings
    3. Automatically selects method, tool, and reranking

    Routes:
    - greetings → No retrieval
    - factual_qa → Hybrid search
    - comparison → Multi-query if available
    - analytical_qa → Multi-query + reranking
    - definition → Hybrid search
    - procedural → Dense search

    Example:
        route_retriever = RouteRetriever()

        # Factual query → Hybrid search
        results = route_retriever.search("What is BERT?")

        # Comparison → Multi-query
        results = route_retriever.search("Compare BERT and GPT-3")

        # Greeting → No retrieval
        results = route_retriever.search("Hello!")
    """

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
        use_cloud: bool = True,
        qdrant_client: Optional[QdrantClient] = None
    ):
        """
        Initialize route-driven retriever.

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

        # Lazy loading for retrieval methods
        self._hybrid_retriever = None

        print("✅ Route-driven retriever initialized")

    def _get_hybrid_retriever(self) -> HybridRetriever:
        """Get or create hybrid retriever (lazy loading)."""
        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetriever(
                strategy=self.strategy,
                use_cloud=False,
                qdrant_client=self.qdrant_client
            )
        return self._hybrid_retriever

    def search(
        self,
        query: str,
        verbose: bool = False
    ) -> List[Document]:
        """
        Route-driven search with automatic method selection.

        Args:
            query: Search query
            verbose: Print routing and configuration details

        Returns:
            Retrieved documents (empty list if no retrieval needed)

        Example:
            # Automatic routing
            results = route_retriever.search("What is BERT?")

            # With details
            results = route_retriever.search("What is BERT?", verbose=True)
        """
        # Step 1: Route the query
        route_result = route_with_complexity_analysis(query)
        route = route_result["route"]
        complexity = route_result["complexity"]

        if verbose:
            print(f"\n🔍 Query: {query[:80]}...")
            print(f"\n🎯 Route Analysis:")
            print(f"   Route: {route['route_name']}")
            print(f"   Requires Retrieval: {route['requires_retrieval']}")
            print(f"   Method: {route.get('retrieval_method', 'N/A')}")
            print(f"   Tool: {route.get('tool', 'N/A')}")
            print(f"   Complexity: {complexity['complexity']} ({complexity['score']}/5)")
            print(f"   Use Reranking: {route.get('use_reranking', False)}\n")

        # Step 2: Early exit if no retrieval needed
        if not route.get("requires_retrieval", True):
            if verbose:
                print("⏭️  No retrieval needed (greeting/casual)\n")
            return []

        # Step 3: Get retrieval method
        method = route.get("retrieval_method", "dense_search")

        # Step 4: Execute retrieval based on method
        if method == "hybrid_search":
            # Hybrid search (BM25 + Dense)
            if verbose:
                print("🔄 Using hybrid search (BM25 + Dense)")

            hybrid = self._get_hybrid_retriever()
            k = 5 if complexity["complexity"] == "simple" else 8
            results = hybrid.search(query, k=k)

        elif method == "multi_query" and MULTI_QUERY_AVAILABLE:
            # Multi-query retrieval
            if verbose:
                print("🔄 Using multi-query retrieval")

            # Generate query variants
            variants = expand_query(query, num_variants=3)

            if verbose:
                print(f"   Generated {len(variants)} query variants")

            # Retrieve with each variant
            all_results = []
            for variant in variants:
                results_variant = self.vector_store.similarity_search(variant, k=10)
                all_results.append(results_variant)

            # Merge with RRF
            k = 5 if complexity["complexity"] == "simple" else 8
            results = reciprocal_rank_fusion(all_results, k=60)[:k]

        elif method == "multi_query" and not MULTI_QUERY_AVAILABLE:
            # Fallback to hybrid if multi-query not available
            if verbose:
                print("⚠️  Multi-query not available, using hybrid search")

            hybrid = self._get_hybrid_retriever()
            k = 5 if complexity["complexity"] == "simple" else 8
            results = hybrid.search(query, k=k)

        else:
            # Default: Dense search
            if verbose:
                print("🔄 Using dense search")

            k = 3 if complexity["complexity"] == "simple" else 5
            results = self.vector_store.similarity_search(query, k=k)

        if verbose:
            print(f"✅ Retrieved {len(results)} documents\n")

        return results

    def get_route_info(self, query: str) -> Dict[str, Any]:
        """
        Get routing information without retrieval.

        Useful for understanding how a query would be routed!

        Args:
            query: Search query

        Returns:
            Route and complexity information

        Example:
            info = route_retriever.get_route_info("What is BERT?")
            print(f"Route: {info['route']['route_name']}")
            print(f"Method: {info['route']['retrieval_method']}")
        """
        return route_with_complexity_analysis(query)

    def compare_routes(self, queries: List[str]) -> Dict[str, Dict]:
        """
        Compare how different queries are routed.

        Args:
            queries: List of queries to compare

        Returns:
            Dictionary mapping queries to their routing info

        Example:
            queries = [
                "What is BERT?",
                "Compare BERT and GPT-3",
                "Hello!"
            ]
            comparison = route_retriever.compare_routes(queries)
        """
        comparison = {}
        for query in queries:
            route_info = self.get_route_info(query)
            comparison[query] = {
                "route": route_info["route"]["route_name"],
                "method": route_info["route"].get("retrieval_method", "none"),
                "complexity": route_info["complexity"]["complexity"],
                "requires_retrieval": route_info["route"].get("requires_retrieval", True)
            }
        return comparison

    def close(self):
        """Close Qdrant client connection."""
        self.qdrant_client.close()
        if self._hybrid_retriever:
            self._hybrid_retriever.close()


# Simple helper function
def route_search(
    query: str,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
    verbose: bool = False
) -> List[Document]:
    """
    Quick route-driven search - no class initialization needed.

    Args:
        query: Search query
        strategy: Chunking strategy
        verbose: Print routing details

    Returns:
        Retrieved documents

    Example:
        from mid_retrieval.route_driven_retrieval import route_search

        results = route_search("What is BERT?", verbose=True)
    """
    retriever = RouteRetriever(strategy=strategy)
    results = retriever.search(query, verbose=verbose)
    retriever.close()
    return results


# Example usage
if __name__ == "__main__":
    # Initialize
    route_retriever = RouteRetriever(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    # Test different query types
    test_queries = [
        "Hello!",  # Greeting
        "What is BERT?",  # Factual
        "Compare BERT and GPT-3",  # Comparison
        "How does attention mechanism work?",  # Analytical
        "Define transformer architecture",  # Definition
    ]

    print("\n" + "="*60)
    print("ROUTE-DRIVEN RETRIEVAL DEMO")
    print("="*60)

    for query in test_queries:
        print(f"\n{'='*60}")
        results = route_retriever.search(query, verbose=True)
        print(f"Final: {len(results)} documents retrieved")

    # Show routing comparison
    print("\n" + "="*60)
    print("ROUTING COMPARISON")
    print("="*60)

    comparison = route_retriever.compare_routes(test_queries)
    for query, info in comparison.items():
        print(f"\nQuery: {query[:50]}...")
        print(f"  Route: {info['route']}")
        print(f"  Method: {info['method']}")
        print(f"  Complexity: {info['complexity']}")
        print(f"  Retrieval: {info['requires_retrieval']}")

    route_retriever.close()
