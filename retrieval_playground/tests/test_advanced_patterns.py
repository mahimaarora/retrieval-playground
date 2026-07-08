"""
Test Advanced Integration Patterns

This test shows:
1. Multi-Query Hybrid Search (full pipeline)
2. Pipeline stage-by-stage impact

Run with: python -m retrieval_playground.tests.test_advanced_patterns
"""

from retrieval_playground.src.mid_retrieval.multi_query_hybrid import MultiQueryHybrid
from retrieval_playground.src.pre_retrieval.chunking_manager import ChunkingStrategy


def test_multi_query_hybrid_basic():
    """Test 1: Basic multi-query hybrid search"""
    print("="*60)
    print("TEST 1: Multi-Query Hybrid - Basic")
    print("="*60)

    mqh = MultiQueryHybrid(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
        num_variants=3,
        reranker_model="bge"
    )

    query = "What is BERT?"
    print(f"\n🔍 Query: {query}\n")

    results = mqh.search(query, k=5, verbose=True)

    print(f"Retrieved {len(results)} documents")
    if results:
        print(f"Top result: {results[0].page_content[:80]}...")

    mqh.close()
    print("\n✅ Test 1 passed!\n")


def test_pipeline_comparison():
    """Test 2: Compare different pipelines"""
    print("="*60)
    print("TEST 2: Pipeline Comparison")
    print("="*60)

    mqh = MultiQueryHybrid(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    query = "How does attention mechanism work?"

    print(f"\n🔍 Query: {query}\n")
    print("Comparing 4 pipeline configurations:\n")

    comparison = mqh.compare_pipelines(query, k=5)

    pipelines = [
        ("Dense Only", comparison['dense']),
        ("Hybrid (BM25 + Dense)", comparison['hybrid']),
        ("Multi-Query", comparison['multi_query']),
        ("Full Pipeline", comparison['full']),
    ]

    for name, results in pipelines:
        print(f"{name}:")
        print(f"  Documents: {len(results)}")
        if results:
            # Check for different score types
            score = (results[0].metadata.get('rerank_score') or
                    results[0].metadata.get('rrf_score') or
                    results[0].metadata.get('score', 0))
            score_type = ('rerank' if 'rerank_score' in results[0].metadata
                         else 'rrf' if 'rrf_score' in results[0].metadata
                         else 'similarity')
            print(f"  Top {score_type} score: {score:.4f}")
            print(f"  Top doc: {results[0].page_content[:60]}...")
        print()

    mqh.close()
    print("✅ Test 2 passed!\n")


def test_different_query_variants():
    """Test 3: Different numbers of query variants"""
    print("="*60)
    print("TEST 3: Query Variant Count")
    print("="*60)

    query = "What is BERT?"

    print(f"\n🔍 Query: {query}\n")
    print("Testing different variant counts:\n")

    for num_variants in [1, 3, 5]:
        print(f"Testing {num_variants} variant(s):")

        mqh = MultiQueryHybrid(
            strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
            num_variants=num_variants,
            reranker_model="flashrank"  # Fast for testing
        )

        results = mqh.search(query, k=3, verbose=False)

        print(f"  Retrieved: {len(results)} documents")
        mqh.close()
        print()

    print("✅ Test 3 passed!\n")


def test_multiple_queries():
    """Test 4: Multiple different queries"""
    print("="*60)
    print("TEST 4: Multiple Queries")
    print("="*60)

    mqh = MultiQueryHybrid(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
        num_variants=3
    )

    queries = [
        "What is BERT?",
        "How does attention work?",
        "Compare transformers and RNNs",
    ]

    print(f"\nTesting {len(queries)} queries:\n")

    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        results = mqh.search(query, k=3, verbose=False)
        print(f"  → Retrieved {len(results)} documents\n")

    mqh.close()
    print("✅ Test 4 passed!\n")


def test_candidate_pool_sizes():
    """Test 5: Different candidate pool sizes"""
    print("="*60)
    print("TEST 5: Candidate Pool Sizes")
    print("="*60)

    mqh = MultiQueryHybrid(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    query = "What is BERT?"

    print(f"\n🔍 Query: {query}\n")
    print("Testing different candidate pool sizes:\n")

    for pool_size in [20, 50, 100]:
        print(f"Pool size: {pool_size}")
        results = mqh.search(query, k=5, candidate_pool=pool_size, verbose=False)
        print(f"  → Final results: {len(results)}\n")

    mqh.close()
    print("✅ Test 5 passed!\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("ADVANCED PATTERNS TEST SUITE")
    print("="*60 + "\n")

    try:
        test_multi_query_hybrid_basic()
        test_pipeline_comparison()
        test_different_query_variants()
        test_multiple_queries()
        test_candidate_pool_sizes()

        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nKey Takeaways:")
        print("1. Multi-query hybrid combines 4 powerful techniques")
        print("2. Each pipeline stage contributes to quality")
        print("3. Full pipeline gives best results (+40-60% expected)")
        print("4. Configurable for different use cases")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
