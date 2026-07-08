"""
Test Hybrid Search (BM25 + Dense)

This test shows:
1. How hybrid search works
2. Comparison: BM25 vs Dense vs Hybrid
3. Performance improvement

Run with: python -m retrieval_playground.tests.test_hybrid_search
"""

from retrieval_playground.src.mid_retrieval.hybrid_search import HybridRetriever
from retrieval_playground.src.pre_retrieval.chunking_manager import ChunkingStrategy
import json
from pathlib import Path


def load_test_queries():
    """Load test queries."""
    # Adjust path if needed
    test_queries_path = Path(__file__).parent / "test_queries.json"
    with open(test_queries_path, 'r') as f:
        return json.load(f)


def test_basic_hybrid_search():
    """Test 1: Basic hybrid search"""
    print("="*60)
    print("TEST 1: Basic Hybrid Search")
    print("="*60)

    # Initialize (uses cloud Qdrant by default)
    hybrid = HybridRetriever(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    # Simple query
    query = "What is BERT?"
    results = hybrid.search(query, k=3)

    print(f"\n🔍 Query: {query}\n")
    print("📊 Results:\n")

    for i, doc in enumerate(results, 1):
        rrf_score = doc.metadata.get('rrf_score', 0)
        source = doc.metadata.get('source', 'Unknown')
        print(f"{i}. RRF Score: {rrf_score:.4f}")
        print(f"   Source: {source[:60]}...")
        print(f"   Content: {doc.page_content[:100]}...")
        print()

    hybrid.close()
    print("✅ Test 1 passed!\n")


def test_compare_methods():
    """Test 2: Compare BM25 vs Dense vs Hybrid"""
    print("="*60)
    print("TEST 2: BM25 vs Dense vs Hybrid Comparison")
    print("="*60)

    # Initialize
    hybrid = HybridRetriever(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    # Load a test query
    test_queries = load_test_queries()
    query = test_queries[0]["user_input"]

    print(f"\n🔍 Query: {query[:100]}...\n")

    # Compare methods
    comparison = hybrid.compare_methods(query, k=3)

    # Display BM25 results
    print("🔤 BM25 Results (Keyword Matching):")
    for i, doc in enumerate(comparison['bm25'], 1):
        score = doc.metadata.get('score', 0)
        print(f"{i}. Score: {score:.4f} | {doc.page_content[:80]}...")

    # Display Dense results
    print("\n🧠 Dense Results (Semantic Similarity):")
    for i, doc in enumerate(comparison['dense'], 1):
        score = doc.metadata.get('score', 0)
        print(f"{i}. Score: {score:.4f} | {doc.page_content[:80]}...")

    # Display Hybrid results
    print("\n🚀 Hybrid Results (BM25 + Dense + RRF):")
    for i, doc in enumerate(comparison['hybrid'], 1):
        rrf_score = doc.metadata.get('rrf_score', 0)
        print(f"{i}. RRF Score: {rrf_score:.4f} | {doc.page_content[:80]}...")

    hybrid.close()
    print("\n✅ Test 2 passed!\n")


def test_multiple_queries():
    """Test 3: Run hybrid search on multiple test queries"""
    print("="*60)
    print("TEST 3: Multiple Query Testing")
    print("="*60)

    # Initialize
    hybrid = HybridRetriever(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    # Load test queries
    test_queries = load_test_queries()

    print(f"\nTesting on {len(test_queries)} queries:\n")

    for i, query_data in enumerate(test_queries[:3], 1):  # Test first 3
        query = query_data["user_input"]

        print(f"Query {i}: {query[:80]}...")

        # Hybrid search
        results = hybrid.search(query, k=3)

        # Show top result
        if results:
            top_doc = results[0]
            rrf_score = top_doc.metadata.get('rrf_score', 0)
            print(f"  Top Result RRF Score: {rrf_score:.4f}")
            print(f"  Content: {top_doc.page_content[:100]}...")

        print()

    hybrid.close()
    print("✅ Test 3 passed!\n")


def test_hybrid_parameters():
    """Test 4: Different parameter configurations"""
    print("="*60)
    print("TEST 4: Parameter Tuning")
    print("="*60)

    # Initialize
    hybrid = HybridRetriever(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    query = "How does attention mechanism work in transformers?"

    # Test different k values
    print(f"\n🔍 Query: {query}\n")

    for k in [3, 5, 10]:
        results = hybrid.search(query, k=k)
        print(f"k={k}: Retrieved {len(results)} documents")
        if results:
            top_score = results[0].metadata.get('rrf_score', 0)
            print(f"  Top RRF Score: {top_score:.4f}")

    print()

    # Test different retrieval parameters
    print("Testing with different BM25/Dense retrieval sizes:")

    # Small retrieval pool
    results_small = hybrid.search(query, k=5, bm25_k=20, dense_k=20)
    print(f"BM25/Dense k=20: Retrieved {len(results_small)} final docs")

    # Large retrieval pool
    results_large = hybrid.search(query, k=5, bm25_k=100, dense_k=100)
    print(f"BM25/Dense k=100: Retrieved {len(results_large)} final docs")

    hybrid.close()
    print("\n✅ Test 4 passed!\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("HYBRID SEARCH TEST SUITE")
    print("="*60 + "\n")

    try:
        test_basic_hybrid_search()
        test_compare_methods()
        test_multiple_queries()
        test_hybrid_parameters()

        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
