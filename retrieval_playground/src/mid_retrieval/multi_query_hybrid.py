"""
Multi-Query Hybrid Search: Complete 4-stage pipeline

Combines:
1. Multi-query generation (expand query into variants)
2. Hybrid search (BM25 + Dense for each variant)
3. Reciprocal Rank Fusion (merge all results)
4. Reranking (cross-encoder for final precision)
"""

from typing import List
from langchain_core.documents import Document

from retrieval_playground.src.pre_retrieval.query_rephrasing import expand_query
from retrieval_playground.src.mid_retrieval.hybrid_search import HybridRetriever
from retrieval_playground.src.mid_retrieval.reranking import Reranker


class MultiQueryHybrid:
    """
    4-stage retrieval pipeline combining multiple techniques.

    Pipeline:
    1. Multi-query: Generate N query variants (default: 3)
    2. Hybrid search: BM25 + Dense for each variant
    3. RRF fusion: Merge all results
    4. Reranking: Cross-encoder for top-k

    Example:
        mqh = MultiQueryHybrid(collection_name="hybrid")
        results = mqh.search("What is BERT?", k=5)
    """

    def __init__(
        self,
        collection_name: str = "hybrid",
        reranker_collection: str = "recursive_character",
        num_variants: int = 3,
        use_cloud: bool = True
    ):
        """
        Args:
            collection_name: Qdrant collection for hybrid search (must have BM25 + dense vectors)
            reranker_collection: Collection for initial retrieval in reranker (e.g., "recursive_character")
            num_variants: Number of query variants to generate
            use_cloud: Use cloud Qdrant vs local
        """
        self.collection_name = collection_name
        self.num_variants = num_variants

        # Initialize hybrid retriever (for BM25 + Dense)
        self.hybrid_retriever = HybridRetriever(collection_name=collection_name)

        # Initialize reranker (for cross-encoder)
        self.reranker = Reranker(
            collection_name=reranker_collection,
            top_k=100,  # Candidates before reranking
            top_n=5,    # Will be overridden by search() k parameter
            use_cloud=use_cloud
        )

        print(f"✅ Multi-query hybrid initialized ({num_variants} variants, FlashRank reranker)")

    def search(
        self,
        query: str,
        k: int = 5,
        candidate_pool: int = 100
    ) -> List[Document]:
        """
        4-stage retrieval pipeline.

        Args:
            query: Search query
            k: Final number of results
            candidate_pool: Candidates before reranking

        Returns:
            Top-k reranked documents
        """
        # Stage 1: Generate query variants
        variants = expand_query(query, num_variants=self.num_variants)

        # Stage 2: Hybrid search for each variant
        all_results = []
        for variant in variants:
            # Get hybrid results (BM25 + Dense + RRF)
            hybrid_results = self.hybrid_retriever.search(
                variant,
                k=50,
                sparse_k=30,
                dense_k=30,
                rrf_k=60
            )
            all_results.append(hybrid_results)

        # Stage 3: RRF fusion across all variants
        fused_scores = {}
        doc_map = {}

        for results in all_results:
            for rank, doc in enumerate(results):
                doc_id = hash(doc.page_content)

                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                    doc_map[doc_id] = doc

                # RRF: 1 / (k + rank)
                fused_scores[doc_id] += 1 / (60 + rank)

        # Sort by RRF score
        ranked_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        # Top candidates for reranking
        merged = []
        for doc_id, rrf_score in ranked_ids[:candidate_pool]:
            doc = doc_map[doc_id]
            doc.metadata['rrf_score'] = float(rrf_score)
            merged.append(doc)

        if not merged:
            return []

        # Stage 4: Reranking with cross-encoder
        reranked = self.reranker._rerank(merged, query)
        return reranked[:k]

    def compare_pipelines(self, query: str, k: int = 5) -> dict:
        """
        Compare: dense-only vs hybrid vs multi-query vs full pipeline.

        Args:
            query: Search query
            k: Number of results

        Returns:
            Dictionary with results from each pipeline
        """
        # Dense only
        dense = self.hybrid_retriever._dense_search(query, k=k)

        # Hybrid (BM25 + Dense + RRF)
        hybrid_results = self.hybrid_retriever.search(query, k=k)

        # Multi-query + Dense (no hybrid)
        variants = expand_query(query, num_variants=self.num_variants)
        multi_results = []
        for variant in variants:
            results = self.hybrid_retriever._dense_search(variant, k=20)
            multi_results.append(results)

        # RRF across variants
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
        multi_query = [doc_map[doc_id] for doc_id, _ in ranked_ids[:k]]

        # Full pipeline (multi-query + hybrid + reranking)
        full_pipeline = self.search(query, k=k)

        return {
            "dense": dense,
            "hybrid": hybrid_results,
            "multi_query": multi_query,
            "full_pipeline": full_pipeline
        }

    def close(self):
        """Close Qdrant connections."""
        self.hybrid_retriever.close()
        self.reranker.close()


if __name__ == "__main__":
    """
    Run: python -m retrieval_playground.src.mid_retrieval.multi_query_hybrid
    """

    mqh = MultiQueryHybrid(
        collection_name="hybrid",
        reranker_collection="recursive_character",
        num_variants=3,
        use_cloud=True
    )

    # Example 1
    print("\n" + "=" * 80)
    query1 = "What is Agent Laboratory?"
    print(f"Query: {query1}")
    print("=" * 80)

    comparison1 = mqh.compare_pipelines(query1, k=3)

    print("\nDENSE (semantic only):")
    for i, doc in enumerate(comparison1['dense'], 1):
        print(f"  {i}. {doc.page_content[:120]}...\n")

    print("HYBRID (BM25 + Dense + RRF):")
    for i, doc in enumerate(comparison1['hybrid'], 1):
        print(f"  {i}. {doc.page_content[:120]}...\n")

    print("MULTI-QUERY (3 variants + Dense + RRF):")
    for i, doc in enumerate(comparison1['multi_query'], 1):
        print(f"  {i}. {doc.page_content[:120]}...\n")

    print("FULL PIPELINE (Multi-query + Hybrid + Reranking):")
    for i, doc in enumerate(comparison1['full_pipeline'], 1):
        score = doc.metadata.get('rerank_score', 0)
        print(f"  {i}. Score: {score:.4f}")
        print(f"     {doc.page_content[:120]}...\n")

    # Example 2
    print("=" * 80)
    query2 = "How do AI agents improve scientific research workflows?"
    print(f"Query: {query2}")
    print("=" * 80)

    comparison2 = mqh.compare_pipelines(query2, k=3)

    print("\nDENSE (semantic only):")
    for i, doc in enumerate(comparison2['dense'], 1):
        print(f"  {i}. {doc.page_content[:120]}...\n")

    print("HYBRID (BM25 + Dense + RRF):")
    for i, doc in enumerate(comparison2['hybrid'], 1):
        print(f"  {i}. {doc.page_content[:120]}...\n")

    print("MULTI-QUERY (3 variants + Dense + RRF):")
    for i, doc in enumerate(comparison2['multi_query'], 1):
        print(f"  {i}. {doc.page_content[:120]}...\n")

    print("FULL PIPELINE (Multi-query + Hybrid + Reranking):")
    for i, doc in enumerate(comparison2['full_pipeline'], 1):
        score = doc.metadata.get('rerank_score', 0)
        print(f"  {i}. Score: {score:.4f}")
        print(f"     {doc.page_content[:120]}...\n")

    print("=" * 80 + "\n")
    mqh.close()
