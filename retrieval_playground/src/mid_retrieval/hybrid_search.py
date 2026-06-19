"""
Hybrid Search: Combine BM25 (keyword) + Dense (semantic) retrieval

Why Hybrid?
- BM25: Great for exact keywords, acronyms, rare terms
- Dense: Great for semantic meaning, paraphrases
- Combined: Best of both worlds! +15-25% improvement

Simple to use:
    hybrid = HybridRetriever()
    results = hybrid.search("your query here")
"""

from typing import List, Dict, Any, Optional
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np

from retrieval_playground.utils import config, constants
from retrieval_playground.utils.model_manager import model_manager
from retrieval_playground.src.pre_retrieval.chunking_manager import ChunkingStrategy


class HybridRetriever:
    """
    Simple hybrid search combining BM25 + Dense retrieval.

    How it works:
    1. BM25 finds documents with exact keyword matches
    2. Dense finds documents with similar meaning
    3. Reciprocal Rank Fusion (RRF) merges the results

    Example:
        hybrid = HybridRetriever()
        results = hybrid.search("What is BERT?", k=5)
    """

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
        use_cloud: bool = True,
        qdrant_client: Optional[QdrantClient] = None
    ):
        """
        Initialize hybrid retriever.

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

        # Setup dense retriever (vector search)
        self.embeddings = model_manager.get_embeddings()
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=strategy.value,
            embedding=self.embeddings
        )

        # Setup BM25 index (keyword search)
        self._build_bm25_index()

        print("✅ Hybrid retriever initialized (BM25 + Dense)")

    def _build_bm25_index(self):
        """Build BM25 index from all documents in vector store."""
        print("Building BM25 index...")

        # Get all documents from vector store
        # Note: This is simple but loads everything into memory
        # For production with millions of docs, use a persistent BM25 index
        collection_name = self.strategy.value
        scroll_result = self.qdrant_client.scroll(
            collection_name=collection_name,
            limit=10000,  # Adjust based on your dataset size
            with_payload=True,
            with_vectors=False
        )

        points = scroll_result[0]

        # Extract documents and metadata
        self.documents = []
        self.doc_ids = []
        self.doc_metadata = []

        for point in points:
            content = point.payload.get("page_content", "")
            metadata = point.payload.get("metadata", {})

            self.documents.append(content)
            self.doc_ids.append(str(point.id))
            self.doc_metadata.append(metadata)

        # Tokenize documents for BM25
        # Simple word-level tokenization
        tokenized_docs = [doc.lower().split() for doc in self.documents]

        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)

        print(f"✅ BM25 index built with {len(self.documents)} documents")

    def _bm25_search(self, query: str, k: int = 50) -> List[Document]:
        """
        BM25 keyword search.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of Documents with BM25 scores
        """
        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]

        # Create Document objects with scores
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include docs with positive scores
                doc = Document(
                    page_content=self.documents[idx],
                    metadata={
                        **self.doc_metadata[idx],
                        "score": float(scores[idx]),
                        "search_type": "bm25"
                    }
                )
                results.append(doc)

        return results

    def _dense_search(self, query: str, k: int = 50) -> List[Document]:
        """
        Dense semantic search using embeddings.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of Documents with similarity scores
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)

        # Convert to Document objects with consistent metadata
        docs = []
        for doc, score in results:
            doc.metadata["score"] = float(score)
            doc.metadata["search_type"] = "dense"
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
                # Use content as key (simple approach)
                # For production, use document ID
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
        bm25_k: int = 50,
        dense_k: int = 50,
        rrf_k: int = 60
    ) -> List[Document]:
        """
        Hybrid search: BM25 + Dense + RRF.

        Args:
            query: Search query
            k: Number of final results to return
            bm25_k: Number of BM25 results to retrieve
            dense_k: Number of dense results to retrieve
            rrf_k: RRF constant (higher = more weight on lower ranks)

        Returns:
            Top-k merged results

        Example:
            results = hybrid.search("What is BERT?", k=5)
            for doc in results:
                print(f"Score: {doc.metadata['rrf_score']:.3f}")
                print(f"Content: {doc.page_content[:100]}...")
        """
        # Step 1: BM25 search (keyword matching)
        bm25_results = self._bm25_search(query, k=bm25_k)

        # Step 2: Dense search (semantic similarity)
        dense_results = self._dense_search(query, k=dense_k)

        # Step 3: Reciprocal Rank Fusion
        merged = self._reciprocal_rank_fusion(
            [bm25_results, dense_results],
            k=rrf_k
        )

        # Step 4: Return top-k
        return merged[:k]

    def compare_methods(self, query: str, k: int = 5) -> Dict[str, List[Document]]:
        """
        Compare BM25 vs Dense vs Hybrid side-by-side.

        Useful for understanding how each method works!

        Args:
            query: Search query
            k: Number of results per method

        Returns:
            Dictionary with results from each method

        Example:
            results = hybrid.compare_methods("What is BERT?")
            print("BM25 results:", len(results['bm25']))
            print("Dense results:", len(results['dense']))
            print("Hybrid results:", len(results['hybrid']))
        """
        # Get results from each method
        bm25_results = self._bm25_search(query, k=k)
        dense_results = self._dense_search(query, k=k)
        hybrid_results = self.search(query, k=k)

        return {
            "bm25": bm25_results,
            "dense": dense_results,
            "hybrid": hybrid_results
        }

    def close(self):
        """Close Qdrant client connection."""
        self.qdrant_client.close()


# Simple example usage
if __name__ == "__main__":
    # Initialize
    hybrid = HybridRetriever(strategy=ChunkingStrategy.RECURSIVE_CHARACTER, use_cloud=False)

    # Search
    query = "What is BERT?"
    results = hybrid.search(query, k=3)

    print(f"\n🔍 Query: {query}\n")
    print("📊 Hybrid Search Results:\n")

    for i, doc in enumerate(results, 1):
        print(f"{i}. RRF Score: {doc.metadata.get('rrf_score', 0):.3f}")
        print(f"   Content: {doc.page_content[:100]}...")
        print()

    hybrid.close()
