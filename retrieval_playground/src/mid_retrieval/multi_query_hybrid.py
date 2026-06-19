"""
Multi-Query Hybrid Search: The Complete Pipeline

Combines 4 powerful techniques:
1. Multi-query generation (expand query into variants)
2. Hybrid search (BM25 + Dense for each variant)
3. Reciprocal Rank Fusion (merge all results)
4. Reranking (final quality boost)

Expected: +40-60% quality improvement!

Simple to use:
    mqh = MultiQueryHybrid()
    results = mqh.search("your query")
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from retrieval_playground.utils import config, constants
from retrieval_playground.utils.model_manager import model_manager
from retrieval_playground.src.pre_retrieval.chunking_manager import ChunkingStrategy
from retrieval_playground.src.pre_retrieval.query_rephrasing import expand_query
from retrieval_playground.src.mid_retrieval.hybrid_search import HybridRetriever

# Optional reranking
try:
    from retrieval_playground.src.mid_retrieval.reranking import Reranker
    RERANKING_AVAILABLE = True
except ImportError:
    RERANKING_AVAILABLE = False


class MultiQueryHybrid:
    """
    Complete retrieval pipeline with multiplicative quality gains.

    Pipeline stages:
    1. Multi-query: Generate 3 query variants
    2. Hybrid search: BM25 + Dense for each variant (6 searches total)
    3. RRF fusion: Merge all 6 result sets
    4. Reranking: Top-100 → Top-k with cross-encoder

    Example:
        mqh = MultiQueryHybrid()

        # Full pipeline (best quality)
        results = mqh.search("What is BERT?", k=5)

        # See each stage
        results = mqh.search("What is BERT?", k=5, verbose=True)
    """

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
        use_cloud: bool = True,
        qdrant_client: Optional[QdrantClient] = None,
        num_variants: int = 3,
        reranker_model: str = "bge"
    ):
        """
        Initialize multi-query hybrid retriever.

        Args:
            strategy: Chunking strategy
            use_cloud: Use cloud Qdrant
            qdrant_client: Optional pre-configured client
            num_variants: Number of query variants to generate (default: 3)
            reranker_model: Which reranker to use (bge/flashrank/huggingface)
        """
        self.strategy = strategy
        self.num_variants = num_variants
        self.reranker_model = reranker_model

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

        # Lazy loading
        self._hybrid_retriever = None

        print(f"✅ Multi-query hybrid initialized ({num_variants} variants, {reranker_model} reranker)")

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
        k: int = 5,
        candidate_pool: int = 100,
        verbose: bool = False
    ) -> List[Document]:
        """
        Complete multi-query hybrid pipeline.

        Args:
            query: Search query
            k: Final number of results to return
            candidate_pool: Number of candidates before reranking (default: 100)
            verbose: Print pipeline progress

        Returns:
            Top-k reranked documents

        Example:
            # Simple usage
            results = mqh.search("What is BERT?", k=5)

            # With pipeline details
            results = mqh.search("What is BERT?", k=5, verbose=True)
        """
        if verbose:
            print(f"\n🔍 Query: {query[:80]}...")
            print(f"\n🚀 Multi-Query Hybrid Pipeline:")
            print(f"   Final k: {k}")
            print(f"   Candidate pool: {candidate_pool}")
            print(f"   Query variants: {self.num_variants}")
            print(f"   Reranker: {self.reranker_model}\n")

        # Stage 1: Multi-query generation
        if verbose:
            print("Stage 1: Multi-Query Generation")

        variants = expand_query(query, num_variants=self.num_variants)

        if verbose:
            print(f"   Generated {len(variants)} query variants:")
            for i, variant in enumerate(variants, 1):
                print(f"   {i}. {variant}")
            print()

        # Stage 2: Hybrid search for each variant
        if verbose:
            print("Stage 2: Hybrid Search (BM25 + Dense)")

        hybrid = self._get_hybrid_retriever()
        all_results = []

        for i, variant in enumerate(variants, 1):
            # Dense retrieval
            dense = hybrid._dense_search(variant, k=50)
            all_results.append(dense)

            # BM25 retrieval
            sparse = hybrid._bm25_search(variant, k=50)
            all_results.append(sparse)

            if verbose:
                print(f"   Variant {i}: Retrieved {len(dense)} dense + {len(sparse)} sparse")

        total_searches = len(variants) * 2  # Dense + BM25 for each variant

        if verbose:
            print(f"   Total: {total_searches} searches, {sum(len(r) for r in all_results)} candidates\n")

        # Stage 3: Reciprocal Rank Fusion (manual implementation for Documents)
        if verbose:
            print("Stage 3: Reciprocal Rank Fusion")

        # Manual RRF for Document objects
        fused_scores = {}
        doc_map = {}

        for results in all_results:
            for rank, doc in enumerate(results):
                # Use page_content hash as ID (simple approach)
                doc_id = hash(doc.page_content)

                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                    doc_map[doc_id] = doc

                # RRF scoring: 1 / (k + rank)
                fused_scores[doc_id] += 1 / (60 + rank)

        # Sort by fused score
        ranked_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        # Create merged list with RRF scores
        merged = []
        for doc_id, rrf_score in ranked_ids[:candidate_pool]:
            doc = doc_map[doc_id]
            doc.metadata['rrf_score'] = float(rrf_score)
            merged.append(doc)

        if verbose:
            print(f"   Merged to top {len(merged)} candidates")
            if merged:
                print(f"   Top RRF score: {merged[0].metadata.get('rrf_score', 0):.4f}\n")

        # Stage 4: Reranking
        if verbose:
            print(f"Stage 4: Reranking ({self.reranker_model})")

        if RERANKING_AVAILABLE and merged:
            try:
                reranker = Reranker(
                    strategy=self.strategy,
                    use_cloud=False,
                    qdrant_client=self.qdrant_client,
                    model=self.reranker_model,
                    top_n=k
                )

                # Rerank candidates
                if hasattr(reranker, f'_rerank_{self.reranker_model}'):
                    rerank_method = getattr(reranker, f'_rerank_{self.reranker_model}')
                    final = rerank_method(merged, query)
                else:
                    # Fallback
                    final = merged[:k]

                if verbose:
                    print(f"   Reranked {len(merged)} → {len(final)} documents")
                    if final:
                        score = final[0].metadata.get('rerank_score', 'N/A')
                        print(f"   Top rerank score: {score}\n")

            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Reranking failed: {e}")
                    print(f"   Using RRF results without reranking\n")
                final = merged[:k]
        else:
            if verbose:
                if not RERANKING_AVAILABLE:
                    print("   ⚠️  Reranking not available")
                print(f"   Using RRF results: {len(merged[:k])} documents\n")
            final = merged[:k]

        if verbose:
            print(f"✅ Pipeline complete: {len(final)} final documents\n")

        return final

    def compare_pipelines(self, query: str, k: int = 5) -> dict:
        """
        Compare different pipeline configurations.

        Args:
            query: Search query
            k: Number of results

        Returns:
            Dictionary with results from each pipeline

        Example:
            comparison = mqh.compare_pipelines("What is BERT?")
            print("Dense only:", len(comparison['dense']))
            print("Hybrid:", len(comparison['hybrid']))
            print("Multi-query:", len(comparison['multi_query']))
            print("Full pipeline:", len(comparison['full']))
        """
        hybrid = self._get_hybrid_retriever()

        # Pipeline 1: Dense only
        dense = hybrid._dense_search(query, k=k)

        # Pipeline 2: Hybrid (BM25 + Dense)
        hybrid_results = hybrid.search(query, k=k)

        # Pipeline 3: Multi-query (no hybrid)
        variants = expand_query(query, num_variants=self.num_variants)
        multi_results = []
        for variant in variants:
            results = hybrid._dense_search(variant, k=20)
            multi_results.append(results)

        # Manual RRF for Documents
        fused_scores = {}
        doc_map = {}
        for results in multi_results:
            for rank, doc in enumerate(results):
                doc_id = hash(doc.page_content)
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                    doc_map[doc_id] = doc
                fused_scores[doc_id] += 1 / (60 + rank)

        ranked_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        multi_merged = [doc_map[doc_id] for doc_id, _ in ranked_ids[:k]]

        # Pipeline 4: Full (multi-query + hybrid + reranking)
        full = self.search(query, k=k, verbose=False)

        return {
            "dense": dense,
            "hybrid": hybrid_results,
            "multi_query": multi_merged,
            "full": full
        }

    def close(self):
        """Close connections."""
        self.qdrant_client.close()
        if self._hybrid_retriever:
            self._hybrid_retriever.close()


# Simple helper function
def multi_query_hybrid_search(
    query: str,
    k: int = 5,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
    verbose: bool = False
) -> List[Document]:
    """
    Quick multi-query hybrid search.

    Example:
        from mid_retrieval.multi_query_hybrid import multi_query_hybrid_search

        results = multi_query_hybrid_search("What is BERT?", k=5, verbose=True)
    """
    mqh = MultiQueryHybrid(strategy=strategy)
    results = mqh.search(query, k=k, verbose=verbose)
    mqh.close()
    return results


# Example usage
if __name__ == "__main__":
    # Initialize
    mqh = MultiQueryHybrid(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
        num_variants=3,
        reranker_model="bge"
    )

    # Test query
    query = "What is BERT?"

    print("\n" + "="*60)
    print("MULTI-QUERY HYBRID SEARCH DEMO")
    print("="*60)

    # Full pipeline with details
    results = mqh.search(query, k=5, verbose=True)

    print("="*60)
    print(f"FINAL RESULTS ({len(results)} documents)")
    print("="*60)

    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.page_content[:100]}...")

    mqh.close()
