"""
Parent-Child Adaptive Retrieval

Uses adaptive threshold-based strategy to balance precision and context.

Chunk Sizes (from parent_child_chunking.py):
- Child chunks: 1536 chars (≈384 tokens) - for precise retrieval
- Parent chunks: 6144 chars (≈1536 tokens) - for rich context

How it works:
1. Search child chunks first (precise matching)
2. Calculate average similarity score of retrieved children
3. Decision logic:
   - If avg_score >= threshold (default 0.7) → Return children (high quality, stay precise)
   - If avg_score < threshold → Expand to parent chunks (low quality, need more context)

This adaptive approach provides:
- Precision when matches are strong (return focused child chunks)
- Context when matches are weak (expand to comprehensive parent chunks)
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from retrieval_playground.utils import config
from retrieval_playground.utils.model_manager import model_manager


class ParentChildRetriever:
    """
    Parent-child retriever with adaptive threshold-based expansion.

    Strategy:
    - Search child chunks (1536 chars ≈ 384 tokens) first for precise matching
    - Calculate average similarity score
    - If avg_score >= threshold → Keep children (high quality, precise)
    - If avg_score < threshold → Expand to parents (6144 chars ≈ 1536 tokens, 4× context)

    This adaptive approach optimizes the precision-context trade-off:
    - Strong matches: Return focused child chunks
    - Weak matches: Expand to parent chunks for more context

    Example:
        retriever = ParentChildRetriever(expansion_threshold=0.7)
        results = retriever.search("What is BERT?", k=5)
        # Returns children if high quality, parents if low quality
    """

    def __init__(
        self,
        expansion_threshold: float = 0.7,
        use_cloud: bool = True
    ):
        """
        Args:
            expansion_threshold: Score threshold for expansion (higher = more strict)
            use_cloud: Use cloud Qdrant vs local
        """
        self.expansion_threshold = expansion_threshold
        self.collection_name = "parent_child"

        # Setup Qdrant client
        if use_cloud:
            self.qdrant_client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_KEY
            )
        else:
            qdrant_path = config.QDRANT_DIR / self.collection_name
            self.qdrant_client = QdrantClient(path=str(qdrant_path))

        # Setup vector store
        self.embeddings = model_manager.get_embeddings()
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )

        print(f"✅ Parent-child retriever initialized (threshold: {expansion_threshold})")

    def _get_parent_chunks(self, child_docs: List[Document]) -> List[Document]:
        """
        Get parent chunks by parent_id.

        Note: Scrolls all points and filters in Python (no index required).
        """
        parent_docs = []

        # Collect unique parent_ids
        parent_ids_needed = set()
        child_to_parent_map = {}
        for child_doc in child_docs:
            parent_id = child_doc.metadata.get("parent_id")
            if parent_id:
                parent_ids_needed.add(parent_id)
                child_to_parent_map[id(child_doc)] = parent_id

        if not parent_ids_needed:
            return child_docs

        # Scroll and find matching parents (no filter, manual Python filtering)
        parent_cache = {}
        offset = None
        max_scroll = 1000  # Safety limit

        while len(parent_cache) < len(parent_ids_needed) and max_scroll > 0:
            results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            points, offset = results

            for point in points:
                metadata = point.payload.get("metadata", {})
                chunk_type = metadata.get("chunk_type")
                parent_id = metadata.get("parent_id")

                # Found a parent we need
                if chunk_type == "parent" and parent_id in parent_ids_needed:
                    parent_cache[parent_id] = Document(
                        page_content=point.payload.get("page_content", ""),
                        metadata=metadata
                    )

            # Break if no more results
            if offset is None:
                break

            max_scroll -= 1

        # Build result list
        for child_doc in child_docs:
            parent_id = child_to_parent_map.get(id(child_doc))
            if parent_id and parent_id in parent_cache:
                parent_docs.append(parent_cache[parent_id])
            else:
                # Fallback to child if parent not found
                parent_docs.append(child_doc)

        return parent_docs

    def search(
        self,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """
        Parent-child search with automatic expansion.

        Args:
            query: Search query
            k: Number of results

        Returns:
            Child or parent documents (automatic decision based on score quality)
        """
        # Step 1: Search child chunks (retrieve more, filter in Python)
        all_results = self.vector_store.similarity_search_with_score(query, k=k*3)

        # Filter for child chunks only
        children = [
            (doc, score) for doc, score in all_results
            if doc.metadata.get("chunk_type") == "child"
        ][:k]

        if not children:
            return []

        # Step 2: Check quality (average score)
        avg_score = sum(score for _, score in children) / len(children)

        # Step 3: Automatic expansion decision
        if avg_score >= self.expansion_threshold:
            # High quality → Keep precise children
            return [doc for doc, _ in children]
        else:
            # Low quality → Expand to parents for context
            child_docs = [doc for doc, _ in children]
            return self._get_parent_chunks(child_docs)

    def compare_strategies(self, query: str, k: int = 3) -> dict:
        """
        Compare children-only vs parents-only vs parent-child auto.

        Args:
            query: Search query
            k: Number of results

        Returns:
            Dictionary with results and metadata from each strategy
        """
        # Get all results and filter in Python (no index needed)
        all_results = self.vector_store.similarity_search_with_score(query, k=k*3)

        # Children only
        children_results = [
            (doc, score) for doc, score in all_results
            if doc.metadata.get("chunk_type") == "child"
        ][:k]
        children_docs = [doc for doc, _ in children_results]
        children_avg = sum(score for _, score in children_results) / len(children_results) if children_results else 0

        # Parents only
        parents_results = [
            (doc, score) for doc, score in all_results
            if doc.metadata.get("chunk_type") == "parent"
        ][:k]
        parents_docs = [doc for doc, _ in parents_results]

        # Parent-child auto (automatic expansion)
        parent_child_auto_docs = self.search(query, k=k)
        decision = "children" if children_avg >= self.expansion_threshold else "parents"

        return {
            "children": children_docs,
            "parents": parents_docs,
            "parent_child_auto": parent_child_auto_docs,
            "avg_score": children_avg,
            "threshold": self.expansion_threshold,
            "decision": decision
        }

    def close(self):
        """Close Qdrant client connection."""
        self.qdrant_client.close()


if __name__ == "__main__":
    """
    Run: python -m retrieval_playground.src.mid_retrieval.parent_child_retrieval
    """

    retriever = ParentChildRetriever(expansion_threshold=0.7, use_cloud=True)

    # Example 1: Common query (likely high quality → children)
    print("\n" + "=" * 80)
    query1 = "What is Agent Laboratory?"
    print(f"Query: {query1}")
    print("=" * 80)

    comparison1 = retriever.compare_strategies(query1, k=3)

    print(f"\nAvg Score: {comparison1['avg_score']:.4f} | Threshold: {comparison1['threshold']}")
    print(f"Decision: {comparison1['decision'].upper()}\n")

    print("CHILDREN (512 tokens, precise):")
    for i, doc in enumerate(comparison1['children'], 1):
        print(f"  {i}. {doc.page_content[:120]}...\n")

    print("PARENTS (2048 tokens, context):")
    for i, doc in enumerate(comparison1['parents'], 1):
        print(f"  {i}. {doc.page_content[:120]}...\n")

    print(f"PARENT_CHILD_AUTO ({comparison1['decision']}):")
    for i, doc in enumerate(comparison1['parent_child_auto'], 1):
        print(f"  {i}. {doc.page_content[:120]}...\n")

    # Example 2: Moderate query (likely high quality → children)
    print("=" * 80)
    query2 = "How do AI agents improve scientific research workflows?"
    print(f"Query: {query2}")
    print("=" * 80)

    comparison2 = retriever.compare_strategies(query2, k=3)

    print(f"\nAvg Score: {comparison2['avg_score']:.4f} | Threshold: {comparison2['threshold']}")
    print(f"Decision: {comparison2['decision'].upper()}\n")

    print("CHILDREN (512 tokens, precise):")
    for i, doc in enumerate(comparison2['children'], 1):
        print(f"  {i}. {doc.page_content[:120]}...\n")

    print("PARENTS (2048 tokens, context):")
    for i, doc in enumerate(comparison2['parents'], 1):
        print(f"  {i}. {doc.page_content[:120]}...\n")

    print(f"PARENT_CHILD_AUTO ({comparison2['decision']}):")
    for i, doc in enumerate(comparison2['parent_child_auto'], 1):
        print(f"  {i}. {doc.page_content[:120]}...\n")

    # Example 3: Obscure/complex query (likely low quality → parents)
    print("=" * 80)
    query3 = "What are the specific mathematical formulations and implementation details of the reward function optimization in multi-agent reinforcement learning systems?"
    print(f"Query: {query3}")
    print("=" * 80)

    comparison3 = retriever.compare_strategies(query3, k=3)

    print(f"\nAvg Score: {comparison3['avg_score']:.4f} | Threshold: {comparison3['threshold']}")
    print(f"Decision: {comparison3['decision'].upper()}\n")

    print("CHILDREN (512 tokens, precise):")
    for i, doc in enumerate(comparison3['children'], 1):
        print(f"  {i}. {doc.page_content[:120]}...\n")

    print("PARENTS (2048 tokens, context):")
    for i, doc in enumerate(comparison3['parents'], 1):
        print(f"  {i}. {doc.page_content[:120]}...\n")

    print(f"PARENT_CHILD_AUTO ({comparison3['decision']}):")
    for i, doc in enumerate(comparison3['parent_child_auto'], 1):
        print(f"  {i}. {doc.page_content[:120]}...\n")

    print("=" * 80 + "\n")
    retriever.close()
