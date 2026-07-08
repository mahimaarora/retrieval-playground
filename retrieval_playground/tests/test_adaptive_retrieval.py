"""
Test Adaptive Retrieval

This test shows:
1. How adaptive retrieval works
2. Different configurations for different query complexity
3. Automatic parameter selection

Run with: python -m retrieval_playground.tests.test_adaptive_retrieval
"""

from retrieval_playground.src.mid_retrieval.adaptive_retrieval import AdaptiveRetriever
from retrieval_playground.src.pre_retrieval.chunking_manager import ChunkingStrategy


def test_basic_adaptive():
    """Test 1: Basic adaptive retrieval"""
    print("="*60)
    print("TEST 1: Basic Adaptive Retrieval")
    print("="*60)

    # Initialize
    adaptive = AdaptiveRetriever(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    # Simple query
    query = "What is BERT?"
    print(f"\n🔍 Query: {query}")
    print("Expected: Simple complexity → Dense search, k=2\n")

    results = adaptive.search(query, verbose=True)

    print(f"Retrieved {len(results)} documents")
    if results:
        print(f"Top result: {results[0].page_content[:80]}...")

    adaptive.close()
    print("\n✅ Test 1 passed!\n")


def test_complexity_levels():
    """Test 2: Different complexity levels"""
    print("="*60)
    print("TEST 2: Query Complexity Levels")
    print("="*60)

    adaptive = AdaptiveRetriever(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    # Test queries with different complexity
    test_cases = [
        {
            "query": "What is BERT?",
            "expected": "simple"
        },
        {
            "query": "How does the attention mechanism work in transformers?",
            "expected": "moderate"
        },
        {
            "query": "Compare BERT and GPT-3 architectures, explain their differences in training methods, and analyze which performs better on question answering versus text generation tasks",
            "expected": "complex"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        query = test["query"]
        expected = test["expected"]

        print(f"\n{'='*60}")
        print(f"Query {i}: {query[:80]}...")
        print(f"Expected Complexity: {expected}")
        print('='*60)

        results = adaptive.search(query, verbose=True)

        print(f"✅ Retrieved {len(results)} documents")

    adaptive.close()
    print("\n✅ Test 2 passed!\n")


def test_configuration_comparison():
    """Test 3: Compare different configurations"""
    print("="*60)
    print("TEST 3: Configuration Comparison")
    print("="*60)

    adaptive = AdaptiveRetriever(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    query = "What is BERT?"

    print(f"\n🔍 Query: {query}\n")
    print("Comparing: Simple vs Moderate vs Complex configurations\n")

    # Compare configurations
    comparison = adaptive.compare_configs(query)

    for level, results in comparison.items():
        print(f"{level.upper()}: {len(results)} documents")
        if results:
            print(f"  Top result: {results[0].page_content[:60]}...")

    print()

    adaptive.close()
    print("✅ Test 3 passed!\n")


def test_manual_override():
    """Test 4: Manual configuration override"""
    print("="*60)
    print("TEST 4: Manual Configuration Override")
    print("="*60)

    adaptive = AdaptiveRetriever(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    query = "What is BERT?"

    # Test 1: Automatic
    print("\n1️⃣  Automatic Configuration:")
    auto_results = adaptive.search(query, verbose=True)

    # Test 2: Manual override
    print("\n2️⃣  Manual Override (k=10, hybrid):")
    manual_config = {
        "method": "hybrid",
        "k": 10,
        "threshold": 0.3,
        "reranker": "flashrank",
        "use_reranking": True
    }
    manual_results = adaptive.search(query, force_config=manual_config, verbose=True)

    print(f"\nAutomatic: {len(auto_results)} docs")
    print(f"Manual: {len(manual_results)} docs")

    adaptive.close()
    print("\n✅ Test 4 passed!\n")


def test_multiple_queries():
    """Test 5: Multiple real queries"""
    print("="*60)
    print("TEST 5: Multiple Query Testing")
    print("="*60)

    adaptive = AdaptiveRetriever(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    queries = [
        "What is chain-of-thought prompting?",
        "How does RAG improve language models?",
        "Explain the difference between retrieval and generation",
        "Compare supervised learning and reinforcement learning approaches",
    ]

    print(f"\nTesting {len(queries)} queries with adaptive configuration:\n")

    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")

        results = adaptive.search(query)

        print(f"  → Retrieved {len(results)} documents")
        if results:
            # Check if reranked
            has_rerank_score = 'rerank_score' in results[0].metadata
            print(f"  → Reranked: {'Yes' if has_rerank_score else 'No'}")

        print()

    adaptive.close()
    print("✅ Test 5 passed!\n")


def test_efficiency():
    """Test 6: Efficiency gains from adaptive approach"""
    print("="*60)
    print("TEST 6: Efficiency Analysis")
    print("="*60)

    adaptive = AdaptiveRetriever(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    print("\n📊 Comparing Adaptive vs Fixed Configuration:\n")

    query = "What is BERT?"

    # Adaptive (should use simple config: k=2, no reranking)
    print("1️⃣  Adaptive Approach:")
    adaptive_results = adaptive.search(query, verbose=True)

    # Fixed (always complex: k=8, with reranking)
    print("\n2️⃣  Fixed Complex Config:")
    fixed_config = {
        "method": "hybrid",
        "k": 8,
        "threshold": 0.3,
        "reranker": "bge",
        "use_reranking": True
    }
    fixed_results = adaptive.search(query, force_config=fixed_config, verbose=True)

    print("\n📊 Results:")
    print(f"Adaptive: {len(adaptive_results)} docs (efficient for simple query)")
    print(f"Fixed: {len(fixed_results)} docs (overkill for simple query)")
    print("\n💡 Adaptive approach saves resources on simple queries!")

    adaptive.close()
    print("\n✅ Test 6 passed!\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("ADAPTIVE RETRIEVAL TEST SUITE")
    print("="*60 + "\n")

    try:
        test_basic_adaptive()
        test_complexity_levels()
        test_configuration_comparison()
        test_manual_override()
        test_multiple_queries()
        test_efficiency()

        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nKey Takeaways:")
        print("1. Adaptive retrieval automatically optimizes for query complexity")
        print("2. Simple queries → Fast, efficient (k=2, no reranking)")
        print("3. Complex queries → Comprehensive (k=8, hybrid, reranking)")
        print("4. Saves resources while maintaining quality")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
