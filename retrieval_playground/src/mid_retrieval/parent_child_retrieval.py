"""
Parent-Child Adaptive Expansion

Automatically expands from precise child chunks to broader parent chunks
when retrieval quality is low.

How it works:
1. Search child chunks first (precise, 512 tokens)
2. Check result quality (average score)
3. If low quality → Auto-expand to parent chunks (2048 tokens)
4. If high quality → Keep precise child chunks

Simple to use:
    pc_retriever = ParentChildRetriever()
    results = pc_retriever.search("your query")
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from retrieval_playground.utils import config, constants
from retrieval_playground.utils.model_manager import model_manager
from retrieval_playground.src.pre_retrieval.chunking_manager import ChunkingStrategy


class ParentChildRetriever:
    """
    Smart retriever with adaptive parent-child expansion.

    Strategy:
    - Start with precise child chunks (512 tokens)
    - If quality is good (avg score >= threshold) → Keep children
    - If quality is low (avg score < threshold) → Expand to parents (2048 tokens)

    Benefits:
    - Precision when possible
    - Context when needed
    - Automatic decision based on quality

    Example:
        pc_retriever = ParentChildRetriever()

        # High quality query → Returns precise children
        results = pc_retriever.search("What is BERT?")

        # Low quality query → Auto-expands to parents for more context
        results = pc_retriever.search("Obscure technical detail...")
    """

    def __init__(
        self,
        use_cloud: bool = True,
        qdrant_client: Optional[QdrantClient] = None,
        expansion_threshold: float = 0.7
    ):
        """
        Initialize parent-child retriever.

        Args:
            use_cloud: Use cloud Qdrant (True) or local (False)
            qdrant_client: Optional pre-configured Qdrant client
            expansion_threshold: Score threshold for expansion decision
                                 (higher = more strict, expand less often)
        """
        # Use parent_child chunking strategy
        self.strategy = ChunkingStrategy.PARENT_CHILD
        self.expansion_threshold = expansion_threshold

        # Setup Qdrant client
        if use_cloud:
            self.qdrant_client = QdrantClient(
                url=constants.QDRANT_URL,
                api_key=constants.QDRANT_KEY
            )
        elif qdrant_client is None:
            qdrant_path = config.QDRANT_DIR / self.strategy.value
            self.qdrant_client = QdrantClient(path=str(qdrant_path))
        else:
            self.qdrant_client = qdrant_client

        # Setup vector store
        self.embeddings = model_manager.get_embeddings()
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.strategy.value,
            embedding=self.embeddings
        )

        print(f"✅ Parent-child retriever initialized (threshold: {expansion_threshold})")

    def _get_parent_chunks(self, child_docs: List[Document]) -> List[Document]:
        """
        Get parent chunks for given child documents.

        Args:
            child_docs: List of child documents

        Returns:
            List of parent documents
        """
        parent_docs = []

        for child_doc in child_docs:
            parent_id = child_doc.metadata.get("parent_id")

            if parent_id:
                # Query for parent by ID
                # Note: Qdrant point IDs need to be retrieved differently
                # For now, we'll search by metadata
                try:
                    # Search for parent chunk
                    parent_results = self.vector_store.similarity_search(
                        child_doc.page_content,
                        k=10,
                        filter={"chunk_type": "parent"}
                    )

                    # Find matching parent
                    for parent in parent_results:
                        if parent.metadata.get("chunk_id") == parent_id:
                            parent_docs.append(parent)
                            break
                except Exception as e:
                    # If parent not found, keep child
                    parent_docs.append(child_doc)
            else:
                # No parent_id, keep child
                parent_docs.append(child_doc)

        return parent_docs

    def search(
        self,
        query: str,
        k: int = 5,
        verbose: bool = False
    ) -> List[Document]:
        """
        Adaptive parent-child search.

        Args:
            query: Search query
            k: Number of results to retrieve
            verbose: Print decision details

        Returns:
            Child or parent documents (adaptive)

        Example:
            # Automatic expansion decision
            results = pc_retriever.search("What is BERT?", k=5)

            # With details
            results = pc_retriever.search("What is BERT?", k=5, verbose=True)
        """
        if verbose:
            print(f"\n🔍 Query: {query[:80]}...")
            print(f"\n📊 Parent-Child Adaptive Retrieval:")
            print(f"   Initial k: {k}")
            print(f"   Expansion threshold: {self.expansion_threshold}\n")

        # Step 1: Retrieve child chunks (precise)
        try:
            children = self.vector_store.similarity_search_with_score(
                query,
                k=k,
                filter={"chunk_type": "child"}
            )

            if verbose:
                print(f"Step 1: Retrieved {len(children)} child chunks")

        except Exception as e:
            # If filter fails, try without filter
            if verbose:
                print(f"⚠️  Child filter not available, using all chunks")

            children = self.vector_store.similarity_search_with_score(query, k=k)

        if not children:
            if verbose:
                print("❌ No results found")
            return []

        # Step 2: Check quality (average score)
        avg_score = sum(score for _, score in children) / len(children)

        if verbose:
            print(f"\nStep 2: Quality Analysis")
            print(f"   Average score: {avg_score:.3f}")
            print(f"   Threshold: {self.expansion_threshold}")

            # Show individual scores
            print(f"\n   Individual scores:")
            for i, (doc, score) in enumerate(children, 1):
                print(f"   {i}. Score: {score:.3f} - {doc.page_content[:60]}...")

        # Step 3: Adaptive expansion decision
        if avg_score >= self.expansion_threshold:
            # High quality - keep precise children
            if verbose:
                print(f"\n✅ Decision: KEEP CHILDREN (high quality)")
                print(f"   Avg score {avg_score:.3f} >= {self.expansion_threshold}")
                print(f"   Returning {len(children)} precise child chunks\n")

            return [doc for doc, _ in children]

        else:
            # Low quality - expand to parents for more context
            if verbose:
                print(f"\n🔄 Decision: EXPAND TO PARENTS (low quality)")
                print(f"   Avg score {avg_score:.3f} < {self.expansion_threshold}")
                print(f"   Retrieving parent chunks for more context...")

            child_docs = [doc for doc, _ in children]
            parent_docs = self._get_parent_chunks(child_docs)

            if verbose:
                print(f"\n✅ Retrieved {len(parent_docs)} parent chunks")
                print(f"   Parent chunks provide ~4x more context (2048 vs 512 tokens)\n")

            return parent_docs

    def compare_strategies(self, query: str, k: int = 3) -> dict:
        """
        Compare children-only vs parents-only vs adaptive.

        Useful for understanding the expansion decision!

        Args:
            query: Search query
            k: Number of results

        Returns:
            Dictionary with results from each strategy

        Example:
            comparison = pc_retriever.compare_strategies("What is BERT?")
            print("Children:", len(comparison['children']))
            print("Parents:", len(comparison['parents']))
            print("Adaptive:", len(comparison['adaptive']))
        """
        # Children only
        try:
            children = self.vector_store.similarity_search(
                query, k=k, filter={"chunk_type": "child"}
            )
        except:
            children = []

        # Parents only
        try:
            parents = self.vector_store.similarity_search(
                query, k=k, filter={"chunk_type": "parent"}
            )
        except:
            parents = []

        # Adaptive
        adaptive = self.search(query, k=k, verbose=False)

        return {
            "children": children,
            "parents": parents,
            "adaptive": adaptive
        }

    def close(self):
        """Close Qdrant client connection."""
        self.qdrant_client.close()


# Simple helper function
def parent_child_search(
    query: str,
    k: int = 5,
    expansion_threshold: float = 0.7,
    verbose: bool = False
) -> List[Document]:
    """
    Quick parent-child search - no class initialization needed.

    Args:
        query: Search query
        k: Number of results
        expansion_threshold: Score threshold for expansion
        verbose: Print decision details

    Returns:
        Retrieved documents (children or parents, adaptive)

    Example:
        from mid_retrieval.parent_child_retrieval import parent_child_search

        results = parent_child_search("What is BERT?", verbose=True)
    """
    retriever = ParentChildRetriever(expansion_threshold=expansion_threshold)
    results = retriever.search(query, k=k, verbose=verbose)
    retriever.close()
    return results


# Example usage
if __name__ == "__main__":
    # Initialize
    pc_retriever = ParentChildRetriever(expansion_threshold=0.7)

    # Test queries
    test_queries = [
        "What is BERT?",  # Common query - likely high quality
        "Explain obscure technical implementation detail",  # Likely low quality
    ]

    print("\n" + "="*60)
    print("PARENT-CHILD ADAPTIVE RETRIEVAL DEMO")
    print("="*60)

    for query in test_queries:
        print(f"\n{'='*60}")
        results = pc_retriever.search(query, k=3, verbose=True)
        print(f"Final: {len(results)} documents retrieved")

    pc_retriever.close()
