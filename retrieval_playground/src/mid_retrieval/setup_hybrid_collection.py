"""
Setup Hybrid Collection: Copy from recursive_character + add BM25

Instead of re-chunking PDFs:
1. Copy existing chunks from 'recursive_character' collection
2. Add BM25 sparse vectors to them
3. Create new 'hybrid' collection with both dense + BM25 vectors

Efficient: Reuses existing chunks and embeddings!
"""

from retrieval_playground.utils.collection_manager import create_hybrid_collection


def setup_hybrid_collection(
    source_collection: str = "recursive_character",
    target_collection: str = "hybrid",
    overwrite: bool = False
):
    """
    Create hybrid collection by copying from existing collection and adding BM25.

    Args:
        source_collection: Existing collection to copy from (default: "recursive_character")
        target_collection: New hybrid collection name (default: "hybrid")
        overwrite: If True, recreate target collection

    Example:
        setup_hybrid_collection(source_collection="recursive_character", overwrite=True)
    """
    create_hybrid_collection(
        source_collection=source_collection,
        target_collection=target_collection,
        use_cloud=True,
        overwrite=overwrite
    )


if __name__ == "__main__":
    """
    Run: python -m retrieval_playground.src.mid_retrieval.setup_hybrid_collection
    """

    setup_hybrid_collection(
        source_collection="recursive_character",
        target_collection="hybrid",
        overwrite=True
    )
