"""
Route-Driven Retrieval: Automatic method selection based on query type

Automatically:
- Detects query type (factual, comparison, analytical, greeting, etc.)
- Selects appropriate retrieval method
- Adjusts parameters based on complexity

No manual configuration needed!
"""

from typing import List, Dict, Any
from langchain_core.documents import Document

from retrieval_playground.src.pre_retrieval.routing import route_with_complexity_analysis
from retrieval_playground.src.pre_retrieval.query_rephrasing import expand_query
from retrieval_playground.src.mid_retrieval.hybrid_search import HybridRetriever
from retrieval_playground.utils.model_manager import model_manager


class RouteRetriever:
    """
    Automatic retrieval method selection based on query type.

    Routes:
    - greetings → No retrieval
    - factual_qa → Hybrid search
    - comparison → Multi-query
    - analytical_qa → Multi-query
    - definition → Hybrid search
    - procedural → Hybrid search

    Example:
        retriever = RouteRetriever(collection_name="hybrid")
        results = retriever.search("What is BERT?")
    """

    def __init__(
        self,
        collection_name: str = "hybrid",
        use_cloud: bool = True
    ):
        """
        Args:
            collection_name: Qdrant collection (hybrid collection recommended)
            use_cloud: Use cloud Qdrant vs local
        """
        self.collection_name = collection_name

        # Initialize hybrid retriever (supports both hybrid and dense search)
        self.hybrid_retriever = HybridRetriever(collection_name=collection_name)

        # Get embeddings for dense-only fallback
        self.embeddings = model_manager.get_embeddings()

        print("✅ Route-driven retriever initialized")

    def search(
        self,
        query: str
    ) -> List[Document]:
        """
        Automatic method selection based on query analysis.

        Args:
            query: Search query

        Returns:
            Retrieved documents (empty list if no retrieval needed)
        """
        # Step 1: Analyze query (route + complexity)
        analysis = route_with_complexity_analysis(query)
        route = analysis["route"]
        complexity = analysis["complexity"]

        # Step 2: Early exit for greetings
        if not route["requires_retrieval"]:
            return []

        # Step 3: Get final method and complexity-based k
        method = analysis["final_retrieval_method"]
        k = 5 if complexity["complexity"] == "simple" else 8

        # Step 4: Execute retrieval
        if method == "multi_query":
            # Multi-query: Generate variants + RRF
            variants = expand_query(query, num_variants=3)

            all_results = []
            for variant in variants:
                results = self.hybrid_retriever._dense_search(variant, k=15)
                all_results.append(results)

            # RRF fusion
            fused_scores = {}
            doc_map = {}
            for results in all_results:
                for rank, doc in enumerate(results):
                    doc_id = hash(doc.page_content)
                    if doc_id not in fused_scores:
                        fused_scores[doc_id] = 0
                        doc_map[doc_id] = doc
                    fused_scores[doc_id] += 1 / (60 + rank)

            ranked_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            return [doc_map[doc_id] for doc_id, _ in ranked_ids[:k]]

        else:
            # Default: Hybrid search (BM25 + Dense)
            return self.hybrid_retriever.search(query, k=k)

    def get_route_info(self, query: str) -> Dict[str, Any]:
        """
        Get routing information without executing retrieval.

        Args:
            query: Search query

        Returns:
            Route and complexity analysis
        """
        return route_with_complexity_analysis(query)

    def compare_routes(self, queries: List[str]) -> Dict[str, Dict]:
        """
        Show how different queries are routed.

        Args:
            queries: List of queries to analyze

        Returns:
            Dictionary with routing info for each query
        """
        comparison = {}
        for query in queries:
            analysis = self.get_route_info(query)
            comparison[query] = {
                "route": analysis["route"]["route_name"],
                "method": analysis["final_retrieval_method"],
                "complexity": analysis["complexity"]["complexity"],
                "requires_retrieval": analysis["route"]["requires_retrieval"]
            }
        return comparison

    def close(self):
        """Close Qdrant connections."""
        self.hybrid_retriever.close()


if __name__ == "__main__":
    """
    Run: python -m retrieval_playground.src.mid_retrieval.route_driven_retrieval
    """

    retriever = RouteRetriever(collection_name="hybrid", use_cloud=True)

    # Test queries (one for each route type)
    test_queries = [
        ("Hi there! How are you?", "greetings"),
        ("What is Agent Laboratory?", "factual"),
        ("Analyze the impact of reinforcement learning in scientific research", "analytical"),
        ("Compare PyTorch and JAX for scientific computing", "comparison"),
    ]

    print("\n" + "=" * 80)
    print("ROUTE-DRIVEN RETRIEVAL - 4 Route Examples")
    print("=" * 80)

    # Show routing comparison
    queries_only = [q for q, _ in test_queries]
    comparison = retriever.compare_routes(queries_only)

    for (query, expected_route) in test_queries:
        info = comparison[query]

        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print(f"Expected Route: {expected_route} | Detected: {info['route']}")
        print(f"Method: {info['method']} | Complexity: {info['complexity']}")
        print(f"{'=' * 80}")

        if not info['requires_retrieval']:
            print("Result: No retrieval needed (greeting)")
            print()
            continue

        # Execute retrieval
        results = retriever.search(query)
        print(f"\nRetrieved: {len(results)} documents\n")

        # Show top 2 results
        for i, doc in enumerate(results[:2], 1):
            print(f"{i}. {doc.page_content[:120]}...\n")

    print("=" * 80 + "\n")
    retriever.close()
