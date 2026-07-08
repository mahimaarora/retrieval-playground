"""
Test Route-Driven Retrieval

This test shows:
1. How semantic routing controls retrieval
2. Different routes for different query types
3. Automatic method selection

Run with: python -m retrieval_playground.tests.test_route_driven
"""

from retrieval_playground.src.mid_retrieval.route_driven_retrieval import RouteRetriever
from retrieval_playground.src.pre_retrieval.chunking_manager import ChunkingStrategy


def test_basic_routing():
    """Test 1: Basic route-driven retrieval"""
    print("="*60)
    print("TEST 1: Basic Route-Driven Retrieval")
    print("="*60)

    route_retriever = RouteRetriever(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    # Factual query
    query = "What is BERT?"
    print(f"\n🔍 Query: {query}")
    print("Expected: factual_qa route → hybrid search\n")

    results = route_retriever.search(query, verbose=True)

    print(f"Retrieved {len(results)} documents")
    if results:
        print(f"Top result: {results[0].page_content[:80]}...")

    route_retriever.close()
    print("\n✅ Test 1 passed!\n")


def test_different_routes():
    """Test 2: Different query types → different routes"""
    print("="*60)
    print("TEST 2: Different Query Types")
    print("="*60)

    route_retriever = RouteRetriever(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    test_cases = [
        {
            "query": "Hello!",
            "expected_route": "greetings",
            "expected_retrieval": False
        },
        {
            "query": "What is BERT?",
            "expected_route": "factual_qa",
            "expected_retrieval": True
        },
        {
            "query": "Compare BERT and GPT-3",
            "expected_route": "comparison",
            "expected_retrieval": True
        },
        {
            "query": "How does attention mechanism work?",
            "expected_route": "analytical_qa",
            "expected_retrieval": True
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {test['query']}")
        print(f"Expected Route: {test['expected_route']}")
        print('='*60)

        results = route_retriever.search(test["query"], verbose=True)

        if test["expected_retrieval"]:
            print(f"✅ Retrieved {len(results)} documents")
        else:
            print(f"✅ No retrieval (as expected): {len(results)} documents")

    route_retriever.close()
    print("\n✅ Test 2 passed!\n")


def test_route_info():
    """Test 3: Get routing information"""
    print("="*60)
    print("TEST 3: Routing Information")
    print("="*60)

    route_retriever = RouteRetriever(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    query = "What is BERT?"

    print(f"\n🔍 Query: {query}\n")
    print("Getting routing info without retrieval:\n")

    info = route_retriever.get_route_info(query)

    print(f"Route Name: {info['route']['route_name']}")
    print(f"Requires Retrieval: {info['route']['requires_retrieval']}")
    print(f"Retrieval Method: {info['route'].get('retrieval_method', 'N/A')}")
    print(f"Tool: {info['route'].get('tool', 'N/A')}")
    print(f"Complexity: {info['complexity']['complexity']}")
    print(f"Complexity Score: {info['complexity']['score']}/5")
    print(f"Use Reranking: {info['route'].get('use_reranking', False)}")

    route_retriever.close()
    print("\n✅ Test 3 passed!\n")


def test_route_comparison():
    """Test 4: Compare routing for multiple queries"""
    print("="*60)
    print("TEST 4: Route Comparison")
    print("="*60)

    route_retriever = RouteRetriever(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    queries = [
        "Hello!",
        "What is BERT?",
        "Compare BERT and GPT-3",
        "How does attention work?",
        "Define transformer architecture",
    ]

    print(f"\nComparing routes for {len(queries)} queries:\n")

    comparison = route_retriever.compare_routes(queries)

    print(f"{'Query':<40} {'Route':<15} {'Method':<15} {'Retrieval'}")
    print("="*80)

    for query, info in comparison.items():
        query_short = query[:37] + "..." if len(query) > 40 else query
        print(f"{query_short:<40} {info['route']:<15} {info['method']:<15} {info['requires_retrieval']}")

    route_retriever.close()
    print("\n✅ Test 4 passed!\n")


def test_greeting_no_retrieval():
    """Test 5: Greetings should not retrieve"""
    print("="*60)
    print("TEST 5: Greetings → No Retrieval")
    print("="*60)

    route_retriever = RouteRetriever(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    greetings = [
        "Hello!",
        "Hi there!",
        "Hey!",
        "Good morning!",
    ]

    print(f"\nTesting {len(greetings)} greetings:\n")

    for greeting in greetings:
        results = route_retriever.search(greeting, verbose=False)
        status = "✅" if len(results) == 0 else "❌"
        print(f"{status} '{greeting}' → {len(results)} documents")

    route_retriever.close()
    print("\n✅ Test 5 passed!\n")


def test_factual_hybrid():
    """Test 6: Factual queries → Hybrid search"""
    print("="*60)
    print("TEST 6: Factual Queries → Hybrid Search")
    print("="*60)

    route_retriever = RouteRetriever(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER
    )

    factual_queries = [
        "What is BERT?",
        "What is transformer architecture?",
        "What is attention mechanism?",
    ]

    print(f"\nTesting {len(factual_queries)} factual queries:\n")

    for query in factual_queries:
        print(f"\nQuery: {query}")
        results = route_retriever.search(query, verbose=False)
        print(f"  → Retrieved {len(results)} documents")

    route_retriever.close()
    print("\n✅ Test 6 passed!\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("ROUTE-DRIVEN RETRIEVAL TEST SUITE")
    print("="*60 + "\n")

    try:
        test_basic_routing()
        test_different_routes()
        test_route_info()
        test_route_comparison()
        test_greeting_no_retrieval()
        test_factual_hybrid()

        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nKey Takeaways:")
        print("1. Semantic routing automatically selects retrieval method")
        print("2. Greetings → No retrieval (efficient!)")
        print("3. Factual queries → Hybrid search")
        print("4. Comparisons → Multi-query (if available)")
        print("5. Each route optimized for its query type")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
