"""
Test Multiple Reranking Models

This test shows:
1. How to use different reranker models
2. Performance comparison between models
3. Speed vs quality tradeoffs

Run with: python -m retrieval_playground.tests.test_reranking_models
"""

from retrieval_playground.src.mid_retrieval.reranking import Reranker
from retrieval_playground.src.pre_retrieval.chunking_manager import ChunkingStrategy
import json
from pathlib import Path
import time


def load_test_queries():
    """Load test queries."""
    test_queries_path = Path(__file__).parent / "test_queries.json"
    with open(test_queries_path, 'r') as f:
        return json.load(f)


def test_default_reranker():
    """Test 1: Default HuggingFace reranker"""
    print("="*60)
    print("TEST 1: Default HuggingFace Reranker")
    print("="*60)

    # Initialize with default model
    reranker = Reranker(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
        model="huggingface"  # Default
    )

    # Test query
    query = "What is BERT?"
    results = reranker.retrieve(query)

    print(f"\n🔍 Query: {query}\n")
    print("📊 Results:\n")

    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', 'Unknown')
        print(f"{i}. Source: {source[:60]}...")
        print(f"   Content: {doc.page_content[:100]}...")
        print()

    reranker.close_qdrant_client()
    print("✅ Test 1 passed!\n")


def test_bge_reranker():
    """Test 2: BGE Reranker (best free option)"""
    print("="*60)
    print("TEST 2: BGE Reranker v2-m3 (Best Free)")
    print("="*60)

    try:
        # Initialize with BGE model
        reranker = Reranker(
            strategy=ChunkingStrategy.UNSTRUCTURED,
            use_cloud=False,
            model="bge",
            top_n=3
        )

        # Test query
        query = "How does attention mechanism work?"
        results = reranker.retrieve(query)

        print(f"\n🔍 Query: {query}\n")
        print("📊 BGE Reranker Results:\n")

        for i, doc in enumerate(results, 1):
            score = doc.metadata.get('rerank_score', 0)
            print(f"{i}. Rerank Score: {score:.4f}")
            print(f"   Content: {doc.page_content[:100]}...")
            print()

        reranker.close_qdrant_client()
        print("✅ Test 2 passed!\n")

    except ImportError as e:
        print(f"\n⚠️  Test 2 skipped: {e}")
        print("Install with: pip install sentence-transformers\n")


def test_flashrank_reranker():
    """Test 3: FlashRank (fastest)"""
    print("="*60)
    print("TEST 3: FlashRank Reranker (Fastest)")
    print("="*60)

    try:
        # Initialize with FlashRank
        reranker = Reranker(
            strategy=ChunkingStrategy.UNSTRUCTURED,
            use_cloud=False,
            model="flashrank",
            top_n=3
        )

        # Test query
        query = "What is transformer architecture?"
        results = reranker.retrieve(query)

        print(f"\n🔍 Query: {query}\n")
        print("📊 FlashRank Results:\n")

        for i, doc in enumerate(results, 1):
            score = doc.metadata.get('rerank_score', 0)
            print(f"{i}. Rerank Score: {score:.4f}")
            print(f"   Content: {doc.page_content[:100]}...")
            print()

        reranker.close_qdrant_client()
        print("✅ Test 3 passed!\n")

    except ImportError as e:
        print(f"\n⚠️  Test 3 skipped: {e}")
        print("Install with: pip install flashrank\n")


def test_compare_reranker_models():
    """Test 4: Compare all reranker models side-by-side"""
    print("="*60)
    print("TEST 4: Model Comparison (Speed & Quality)")
    print("="*60)

    # Load test queries
    test_queries = load_test_queries()
    query = test_queries[0]["user_input"]

    print(f"\n🔍 Query: {query[:100]}...\n")

    models = ["huggingface", "bge", "flashrank"]
    results_by_model = {}

    for model in models:
        try:
            print(f"\nTesting {model}...")

            # Initialize reranker
            reranker = Reranker(
                strategy=ChunkingStrategy.UNSTRUCTURED,
                use_cloud=False,
                model=model,
                top_n=3
            )

            # Measure time
            start_time = time.time()
            results = reranker.retrieve(query)
            elapsed = time.time() - start_time

            results_by_model[model] = {
                "results": results,
                "time": elapsed
            }

            print(f"✅ {model}: {elapsed:.3f}s")

            reranker.close_qdrant_client()

        except ImportError as e:
            print(f"⚠️  {model} skipped: {e}")
            results_by_model[model] = None

    # Display comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    for model, data in results_by_model.items():
        if data:
            print(f"\n🎯 {model.upper()}:")
            print(f"   Time: {data['time']:.3f}s")
            if data['results']:
                top_doc = data['results'][0]
                score_key = 'rerank_score' if model != 'huggingface' else 'score'
                score = top_doc.metadata.get(score_key, 'N/A')
                print(f"   Top Score: {score}")
                print(f"   Top Doc: {top_doc.page_content[:80]}...")

    print("\n✅ Test 4 passed!\n")


def test_reranker_with_different_top_n():
    """Test 5: Different top_n values"""
    print("="*60)
    print("TEST 5: Different top_n Values")
    print("="*60)

    query = "What is BERT?"

    print(f"\n🔍 Query: {query}\n")

    for top_n in [1, 3, 5]:
        print(f"\nTesting top_n={top_n}:")

        reranker = Reranker(
            strategy=ChunkingStrategy.UNSTRUCTURED,
            use_cloud=False,
            model="huggingface",
            top_n=top_n
        )

        results = reranker.retrieve(query)

        print(f"  Retrieved {len(results)} documents")
        if results:
            print(f"  Top doc: {results[0].page_content[:80]}...")

        reranker.close_qdrant_client()

    print("\n✅ Test 5 passed!\n")


def test_reranker_evaluation():
    """Test 6: Run existing evaluation with different models"""
    print("="*60)
    print("TEST 6: Evaluation Comparison")
    print("="*60)

    models_to_test = ["huggingface"]  # Start with default

    # Try BGE if available
    try:
        from sentence_transformers import CrossEncoder
        models_to_test.append("bge")
    except ImportError:
        print("\n⚠️  BGE not available (install sentence-transformers)\n")

    # Try FlashRank if available
    try:
        from flashrank import Ranker
        models_to_test.append("flashrank")
    except ImportError:
        print("\n⚠️  FlashRank not available (install flashrank)\n")

    print(f"\nRunning evaluation on {len(models_to_test)} model(s):\n")

    evaluation_results = {}

    for model in models_to_test:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model.upper()}")
        print('='*60)

        reranker = Reranker(
            strategy=ChunkingStrategy.UNSTRUCTURED,
            use_cloud=False,
            model=model,
            top_n=3
        )

        # Run evaluation
        results = reranker.evaluate_reranking(close_qdrant_client=True)
        evaluation_results[model] = results

    # Summary comparison
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    for model, results in evaluation_results.items():
        print(f"\n{model.upper()}:")
        print(f"  Average Score: {results['reranker_avg_score']:.4f}")
        print(f"  Improvement: {results['improvement']:.4f}")
        improvement_pct = (results['improvement'] / results['retriever_avg_score']) * 100
        print(f"  Improvement %: {improvement_pct:.2f}%")

    print("\n✅ Test 6 passed!\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("RERANKING MODELS TEST SUITE")
    print("="*60 + "\n")

    try:
        test_default_reranker()
        test_bge_reranker()
        test_flashrank_reranker()
        test_compare_reranker_models()
        test_reranker_with_different_top_n()

        print("\n" + "="*60)
        print("OPTIONAL: Full Evaluation (Takes longer)")
        print("="*60)

        user_input = input("\nRun full evaluation? (y/n): ")
        if user_input.lower() == 'y':
            test_reranker_evaluation()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
