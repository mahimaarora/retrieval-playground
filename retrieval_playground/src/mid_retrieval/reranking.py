"""
Two-stage retrieval with cross-encoder reranking.

Stage 1: Retrieve top_k candidates using bi-encoder (fast, broad)
Stage 2: Rerank using FlashRank cross-encoder (accurate, precise, ~20ms)
"""

from typing import Dict, List
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from flashrank import RerankRequest

from retrieval_playground.utils import config
from retrieval_playground.utils.model_manager import model_manager
from retrieval_playground.utils.pylogger import get_python_logger


class Reranker:
    """
    Two-stage retrieval: fast embedding search + cross-encoder reranking.

    Example:
        reranker = Reranker(collection_name="recursive_character")
        results = reranker.retrieve("What is PyTorch?")

        # Compare baseline vs reranked
        comparison = reranker.compare_results("What is PyTorch?")
    """

    def __init__(
        self,
        collection_name: str,
        top_k: int = 20,
        top_n: int = 3,
        use_cloud: bool = True
    ):
        """
        Args:
            collection_name: Qdrant collection (e.g., "recursive_character")
            top_k: Candidates from initial retrieval (default: 20)
            top_n: Final results after reranking (default: 3)
            use_cloud: Use cloud Qdrant vs local
        """
        self.logger = get_python_logger(log_level=config.PYTHON_LOG_LEVEL)

        if collection_name not in config.SUPPORTED_COLLECTIONS:
            raise ValueError(
                f"Collection '{collection_name}' not supported. "
                f"Must be one of: {config.SUPPORTED_COLLECTIONS}"
            )

        self.collection_name = collection_name
        self.top_k = top_k
        self.top_n = top_n

        if use_cloud:
            self.qdrant_client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_KEY,
                timeout=600
            )
        else:
            qdrant_path = config.QDRANT_DIR / collection_name
            self.qdrant_client = QdrantClient(path=str(qdrant_path))

        self.embeddings = model_manager.get_embeddings()

        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=self.embeddings
        )

        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k}
        )

        self.reranker = model_manager.get_reranker()

        self.logger.info(
            f"✅ Reranker initialized: {collection_name} "
            f"(top_k: {top_k}, top_n: {top_n})"
        )
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve and rerank documents."""
        initial_docs = self.retriever.invoke(query)
        reranked_docs = self._rerank(initial_docs, query)
        return reranked_docs[:self.top_n]

    def _rerank(self, documents: List[Document], query: str) -> List[Document]:
        """Rerank documents using FlashRank cross-encoder."""
        if not documents:
            return []

        # Prepare passages for FlashRank
        passages = [
            {"id": i, "text": doc.page_content}
            for i, doc in enumerate(documents)
        ]

        # Rerank using FlashRank
        request = RerankRequest(query=query, passages=passages)
        results = self.reranker.rerank(request)

        # Sort by score (descending)
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

        # Map back to documents with scores
        reranked = []
        for result in sorted_results:
            doc = documents[result["id"]]
            doc.metadata["rerank_score"] = float(result["score"])
            doc.metadata["reranker"] = "flashrank"
            reranked.append(doc)

        return reranked
    
    def compare_results(self, query: str) -> Dict:
        """Compare baseline vs reranked results side-by-side."""
        # Get initial results WITH scores
        initial_results = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        initial_docs = [doc for doc, score in initial_results]

        # Extract baseline top-n with scores
        baseline_results = initial_results[:self.top_n]
        baseline_docs = [doc for doc, score in baseline_results]
        baseline_scores = [float(score) for doc, score in baseline_results]

        # Rerank
        reranked_docs = self._rerank(initial_docs, query)
        reranked_topn = reranked_docs[:self.top_n]

        return {
            "query": query,
            "top_k_retrieved": len(initial_docs),
            "baseline": baseline_docs,
            "reranked": reranked_topn,
            "baseline_scores": baseline_scores,
            "reranked_scores": [d.metadata.get("rerank_score", 0) for d in reranked_topn]
        }

    def close(self):
        """Close Qdrant client connection."""
        self.qdrant_client.close()

if __name__ == "__main__":
    """
    Run: python -m retrieval_playground.src.mid_retrieval.reranking
    """

    reranker = Reranker(
        collection_name="recursive_character",
        top_k=20,
        top_n=3,
        use_cloud=True
    )

    # Example 1
    print("\n" + "=" * 80)
    query1 = "What is Agent Laboratory?"
    print(f"Query: {query1}")
    print("=" * 80)

    comparison1 = reranker.compare_results(query1)

    print(f"\nBASELINE (retrieved {comparison1['top_k_retrieved']} candidates):")
    for i, (doc, score) in enumerate(zip(comparison1["baseline"], comparison1["baseline_scores"]), 1):
        print(f"  {i}. Score: {score:.4f}")
        print(f"     {doc.page_content[:150]}...\n")

    print("RERANKED:")
    for i, (doc, score) in enumerate(zip(comparison1["reranked"], comparison1["reranked_scores"]), 1):
        print(f"  {i}. Score: {score:.4f}")
        print(f"     {doc.page_content[:150]}...\n")

    # Example 2
    print("=" * 80)
    query2 = "How do AI agents improve scientific research workflows?"
    print(f"Query: {query2}")
    print("=" * 80)

    comparison2 = reranker.compare_results(query2)

    print(f"\nBASELINE (retrieved {comparison2['top_k_retrieved']} candidates):")
    for i, (doc, score) in enumerate(zip(comparison2["baseline"], comparison2["baseline_scores"]), 1):
        print(f"  {i}. Score: {score:.4f}")
        print(f"     {doc.page_content[:150]}...\n")

    print("RERANKED:")
    for i, (doc, score) in enumerate(zip(comparison2["reranked"], comparison2["reranked_scores"]), 1):
        print(f"  {i}. Score: {score:.4f}")
        print(f"     {doc.page_content[:150]}...\n")

    print("=" * 80 + "\n")
    reranker.close()
    