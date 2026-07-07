"""
Hybrid Search: Combine sparse (BM25) + dense (semantic) retrieval

Uses Qdrant's native BM25 - just pass text, Qdrant handles the rest!

Why Hybrid?
- Sparse (BM25): Exact keywords, acronyms, rare terms (TF-IDF weighted by Qdrant)
- Dense: Semantic meaning, paraphrases
- Combined: Best of both worlds! +15-25% improvement

Prerequisites:
    Run setup_hybrid_collection.py first to create collection

Simple to use:
    hybrid = HybridRetriever(collection_name="hybrid")
    results = hybrid.search("your query here")
"""

from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient, models
from langchain_core.documents import Document

from retrieval_playground.utils import config
from retrieval_playground.utils.model_manager import model_manager


class HybridRetriever:
    """
    Hybrid search using Qdrant's native BM25 + dense vectors.

    How it works:
    1. Sparse search: Qdrant native BM25 (pass text, Qdrant handles tokenization/IDF)
    2. Dense search: Semantic embeddings
    3. RRF merges the results

    Example:
        hybrid = HybridRetriever(collection_name="hybrid")
        results = hybrid.search("What is BERT?", k=5)
    """

    def __init__(
        self,
        collection_name: str = "hybrid"
    ):
        """
        Initialize hybrid retriever.

        Args:
            collection_name: Qdrant collection with sparse + dense vectors
                            (use setup_hybrid_collection.py to create)
        """
        self.collection_name = collection_name

        # Setup Qdrant client
        self.client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_KEY
        )

        # Setup embeddings for dense search
        self.embeddings = model_manager.get_embeddings()

        print(f"✅ Hybrid retriever initialized (collection: '{collection_name}')")

    def _sparse_search(self, query: str, k: int = 15) -> List[Document]:
        """
        Sparse (BM25) search using Qdrant native BM25.

        Just pass text - Qdrant handles tokenization and IDF!

        Args:
            query: Search query text
            k: Number of results to return (default: 15)

        Returns:
            List of Documents with BM25 scores
        """
        # Search using Qdrant's native BM25
        # No manual tokenization needed!
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=models.Document(
                text=query,
                model="Qdrant/bm25"
            ),
            using="bm25",  # Use the BM25 sparse vector
            limit=k,
            with_payload=True
        )

        # Convert to Documents
        docs = []
        for point in results.points:
            doc = Document(
                page_content=point.payload.get("page_content", ""),
                metadata={
                    **point.payload.get("metadata", {}),
                    "score": float(point.score),
                    "search_type": "bm25"
                }
            )
            docs.append(doc)

        return docs

    def _dense_search(self, query: str, k: int = 15) -> List[Document]:
        """
        Dense (semantic) search using Qdrant.

        Args:
            query: Search query
            k: Number of results to return (default: 15)

        Returns:
            List of Documents with scores
        """
        # Create dense query vector
        dense_query = self.embeddings.embed_query(query)

        # Search Qdrant dense vectors
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_query,
            using="dense",
            limit=k,
            with_payload=True
        )

        # Convert to Documents
        docs = []
        for point in results.points:
            doc = Document(
                page_content=point.payload.get("page_content", ""),
                metadata={
                    **point.payload.get("metadata", {}),
                    "score": float(point.score),
                    "search_type": "dense"
                }
            )
            docs.append(doc)

        return docs

    def _reciprocal_rank_fusion(
        self,
        results_lists: List[List[Document]],
        k: int = 60
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion: Merge multiple ranked lists.

        Formula: score(doc) = Σ(1 / (k + rank))

        Documents appearing in multiple lists rank higher!

        Args:
            results_lists: List of ranked document lists
            k: Constant for RRF formula (default: 60)

        Returns:
            Merged and re-ranked document list
        """
        # Track scores for each document (by content hash)
        doc_scores = {}
        doc_objects = {}

        for results in results_lists:
            for rank, doc in enumerate(results):
                # Use content hash as key
                doc_key = hash(doc.page_content)

                # RRF score: 1 / (k + rank)
                score = 1.0 / (k + rank + 1)

                if doc_key in doc_scores:
                    doc_scores[doc_key] += score
                else:
                    doc_scores[doc_key] = score
                    doc_objects[doc_key] = doc

        # Sort by RRF score (descending)
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Create final document list with RRF scores
        merged_docs = []
        for doc_key, rrf_score in sorted_docs:
            doc = doc_objects[doc_key]
            doc.metadata["rrf_score"] = float(rrf_score)
            doc.metadata["search_type"] = "hybrid"
            merged_docs.append(doc)

        return merged_docs

    def search(
        self,
        query: str,
        k: int = 5,
        sparse_k: int = 15,
        dense_k: int = 15,
        rrf_k: int = 60
    ) -> List[Document]:
        """
        Hybrid search: Sparse + Dense + RRF.

        Args:
            query: Search query
            k: Number of final results to return
            sparse_k: Number of sparse results to retrieve (default: 15)
            dense_k: Number of dense results to retrieve (default: 15)
            rrf_k: RRF constant (higher = more weight on lower ranks)

        Returns:
            Top-k merged results

        Example:
            results = hybrid.search("What is BERT?", k=5)
            for doc in results:
                print(f"Score: {doc.metadata['rrf_score']:.3f}")
                print(f"Content: {doc.page_content[:100]}...")
        """
        # Step 1: Sparse search (keyword matching) - in Qdrant
        sparse_results = self._sparse_search(query, k=sparse_k)

        # Step 2: Dense search (semantic) - in Qdrant
        dense_results = self._dense_search(query, k=dense_k)

        # Step 3: Reciprocal Rank Fusion - local merge
        merged = self._reciprocal_rank_fusion(
            [sparse_results, dense_results],
            k=rrf_k
        )

        # Step 4: Return top-k
        return merged[:k]

    def compare_methods(self, query: str, k: int = 5) -> Dict[str, List[Document]]:
        """
        Compare Sparse vs Dense vs Hybrid side-by-side.

        Args:
            query: Search query
            k: Number of results per method

        Returns:
            Dictionary with results from each method

        Example:
            results = hybrid.compare_methods("What is BERT?")
            print("Sparse results:", len(results['sparse']))
            print("Dense results:", len(results['dense']))
            print("Hybrid results:", len(results['hybrid']))
        """
        sparse_results = self._sparse_search(query, k=k)
        dense_results = self._dense_search(query, k=k)
        hybrid_results = self.search(query, k=k)

        return {
            "sparse": sparse_results,
            "dense": dense_results,
            "hybrid": hybrid_results
        }

    def close(self):
        """Close Qdrant client connection."""
        self.client.close()


if __name__ == "__main__":
    """
    Run: python -m retrieval_playground.src.mid_retrieval.hybrid_search
    """

    hybrid = HybridRetriever(collection_name="hybrid")

    # Example 1: Keyword Query
    print("\n" + "=" * 80)
    query1 = "What is AL?"
    print(f"Query: {query1}")
    print("=" * 80)

    comparison1 = hybrid.compare_methods(query1, k=3)

    print("\nSPARSE (BM25 keyword matching):")
    for i, doc in enumerate(comparison1['sparse'][:3], 1):
        print(f"  {i}. Score: {doc.metadata['score']:.3f}")
        print(f"     {doc.page_content[:120]}...\n")

    print("DENSE (semantic embeddings):")
    for i, doc in enumerate(comparison1['dense'][:3], 1):
        print(f"  {i}. Score: {doc.metadata['score']:.3f}")
        print(f"     {doc.page_content[:120]}...\n")

    print("HYBRID (RRF fusion):")
    for i, doc in enumerate(comparison1['hybrid'][:3], 1):
        print(f"  {i}. Score: {doc.metadata['rrf_score']:.3f}")
        print(f"     {doc.page_content[:120]}...\n")

    # Example 2: Semantic Query
    print("=" * 80)
    query2 = "How do AI systems automate scientific experiments?"
    print(f"Query: {query2}")
    print("=" * 80)

    comparison2 = hybrid.compare_methods(query2, k=3)

    print("\nSPARSE (BM25 keyword matching):")
    for i, doc in enumerate(comparison2['sparse'][:3], 1):
        print(f"  {i}. Score: {doc.metadata['score']:.3f}")
        print(f"     {doc.page_content[:120]}...\n")

    print("DENSE (semantic embeddings):")
    for i, doc in enumerate(comparison2['dense'][:3], 1):
        print(f"  {i}. Score: {doc.metadata['score']:.3f}")
        print(f"     {doc.page_content[:120]}...\n")

    print("HYBRID (RRF fusion):")
    for i, doc in enumerate(comparison2['hybrid'][:3], 1):
        print(f"  {i}. Score: {doc.metadata['rrf_score']:.3f}")
        print(f"     {doc.page_content[:120]}...\n")

    # Example 3: Mixed Query
    print("=" * 80)
    query3 = "Compare PyTorch and JAX for deep learning"
    print(f"Query: {query3}")
    print("=" * 80)

    comparison3 = hybrid.compare_methods(query3, k=3)

    print("\nSPARSE (BM25 keyword matching):")
    for i, doc in enumerate(comparison3['sparse'][:3], 1):
        print(f"  {i}. Score: {doc.metadata['score']:.3f}")
        print(f"     {doc.page_content[:120]}...\n")

    print("DENSE (semantic embeddings):")
    for i, doc in enumerate(comparison3['dense'][:3], 1):
        print(f"  {i}. Score: {doc.metadata['score']:.3f}")
        print(f"     {doc.page_content[:120]}...\n")

    print("HYBRID (RRF fusion):")
    for i, doc in enumerate(comparison3['hybrid'][:3], 1):
        print(f"  {i}. Score: {doc.metadata['rrf_score']:.3f}")
        print(f"     {doc.page_content[:120]}...\n")

    print("=" * 80 + "\n")
    hybrid.close()
