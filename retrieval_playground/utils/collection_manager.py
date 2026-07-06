"""
Collection Manager - Unified Qdrant collection management

Handles all Qdrant operations:
- Creating collections (regular and hybrid)
- Overwriting existing collections
- Adding documents to collections
"""

from pathlib import Path
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore

from retrieval_playground.utils import constants, config
from retrieval_playground.utils.pylogger import get_python_logger


class CollectionManager:
    """Manages all Qdrant collection operations."""

    def __init__(self, embeddings):
        """
        Args:
            embeddings: Embedding model instance (from model_manager)
        """
        self.embeddings = embeddings
        self.logger = get_python_logger(log_level=constants.PYTHON_LOG_LEVEL)

    def create_collection(
        self,
        collection_name: str,
        use_cloud: bool = False,
        overwrite: bool = False,
        enable_hybrid: bool = False
    ) -> QdrantClient:
        """
        Create or connect to a Qdrant collection.

        Args:
            collection_name: Name of the collection
            use_cloud: Use cloud Qdrant (True) or local (False)
            overwrite: Delete existing collection and create new one
            enable_hybrid: Add BM25 sparse vectors for hybrid search

        Returns:
            QdrantClient instance
        """
        # Get Qdrant client
        qdrant_client = self._get_qdrant_client(collection_name, use_cloud)

        # Check if collection exists
        collection_exists = self._collection_exists(qdrant_client, collection_name, use_cloud)

        if collection_exists:
            if overwrite:
                self.logger.info(f"🗑️  Deleting existing collection: {collection_name}")
                if use_cloud:
                    qdrant_client.delete_collection(collection_name)
                else:
                    import shutil
                    qdrant_path = config.QDRANT_DIR / collection_name
                    shutil.rmtree(qdrant_path)
                    qdrant_client = self._get_qdrant_client(collection_name, use_cloud)
            else:
                self.logger.info(f"➕ Using existing collection: {collection_name}")
                return qdrant_client

        # Create new collection
        self.logger.info(f"📦 Creating collection: {collection_name}")

        # Get embedding dimension
        embedding_size = len(self.embeddings.embed_query('test'))

        if enable_hybrid:
            # Hybrid collection: dense + BM25
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=embedding_size,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                }
            )
            self.logger.info(f"   ✓ Dense vectors ({embedding_size}-dim) + BM25 sparse vectors")
        else:
            # Regular collection: dense only
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=models.Distance.COSINE
                )
            )
            self.logger.info(f"   ✓ Dense vectors ({embedding_size}-dim)")

        return qdrant_client

    def create_hybrid_from_existing(
        self,
        source_collection: str = "recursive_character",
        target_collection: str = "hybrid",
        use_cloud: bool = True,
        overwrite: bool = False
    ):
        """
        Create hybrid collection by copying from existing collection and adding BM25.

        Args:
            source_collection: Existing collection to copy from
            target_collection: New hybrid collection name
            use_cloud: Use cloud Qdrant
            overwrite: Recreate target collection if it exists
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Creating hybrid collection from: {source_collection}")
        self.logger.info(f"{'='*70}\n")

        # Connect to Qdrant
        qdrant_client = self._get_qdrant_client(source_collection, use_cloud)

        # Check source collection exists
        if not self._collection_exists(qdrant_client, source_collection, use_cloud):
            self.logger.error(f"❌ Source collection '{source_collection}' does not exist!")
            qdrant_client.close()
            return

        # Get source collection info
        source_info = qdrant_client.get_collection(source_collection)
        self.logger.info(f"✓ Source: {source_collection} ({source_info.points_count} points)")

        # Get embedding dimension
        embedding_size = source_info.config.params.vectors.size

        # Check if target exists
        target_exists = self._collection_exists(qdrant_client, target_collection, use_cloud)

        if target_exists and not overwrite:
            self.logger.info(f"✓ Collection '{target_collection}' already exists (use overwrite=True to recreate)")
            qdrant_client.close()
            return

        # Create/recreate target collection
        if target_exists and overwrite:
            self.logger.info(f"🗑️  Deleting existing collection: {target_collection}")
            qdrant_client.delete_collection(target_collection)

        self.logger.info(f"📦 Creating hybrid collection: {target_collection}")
        qdrant_client.create_collection(
            collection_name=target_collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=embedding_size,
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            }
        )
        self.logger.info(f"   ✓ Dense vectors ({embedding_size}-dim) + BM25 sparse vectors")

        # Copy points from source to target with BM25
        self.logger.info(f"\n📤 Copying points and adding BM25 vectors...")

        offset = None
        total_copied = 0
        batch_size = 100

        while True:
            # Get batch from source
            results = qdrant_client.scroll(
                collection_name=source_collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )

            points_batch, offset = results

            if not points_batch:
                break

            # Create new points with dense + BM25
            new_points = []
            for point in points_batch:
                page_content = point.payload.get("page_content", "")
                metadata = point.payload.get("metadata", {})
                dense_vector = point.vector

                # Create BM25 sparse vector
                bm25_vector = models.Document(
                    text=page_content,
                    model="Qdrant/bm25"
                )

                # Create new point
                new_point = models.PointStruct(
                    id=point.id,
                    vector={
                        "dense": dense_vector,
                        "bm25": bm25_vector
                    },
                    payload={
                        "page_content": page_content,
                        "metadata": metadata
                    }
                )
                new_points.append(new_point)

            # Upload batch
            qdrant_client.upsert(collection_name=target_collection, points=new_points)
            total_copied += len(new_points)
            self.logger.info(f"   Copied {total_copied} points...")

            if offset is None:
                break

        self.logger.info(f"\n✅ Complete! Copied {total_copied} points")
        self.logger.info(f"   Collection '{target_collection}' ready for hybrid search")
        self.logger.info(f"{'='*70}\n")

        qdrant_client.close()

    def get_vector_store(
        self,
        collection_name: str,
        use_cloud: bool = False
    ) -> QdrantVectorStore:
        """
        Get LangChain QdrantVectorStore for a collection.

        Args:
            collection_name: Name of the collection
            use_cloud: Use cloud Qdrant

        Returns:
            QdrantVectorStore instance
        """
        qdrant_client = self._get_qdrant_client(collection_name, use_cloud)

        return QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

    def _get_qdrant_client(self, collection_name: str, use_cloud: bool) -> QdrantClient:
        """Get Qdrant client (cloud or local)."""
        if use_cloud:
            return QdrantClient(
                url=constants.QDRANT_URL,
                api_key=constants.QDRANT_KEY,
                timeout=600
            )
        else:
            qdrant_path = config.QDRANT_DIR / collection_name
            return QdrantClient(path=str(qdrant_path))

    def _collection_exists(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        use_cloud: bool
    ) -> bool:
        """Check if collection exists."""
        if use_cloud:
            collections = qdrant_client.get_collections()
            return any(col.name == collection_name for col in collections.collections)
        else:
            qdrant_path = config.QDRANT_DIR / collection_name
            return qdrant_path.exists() and any(qdrant_path.iterdir())


# Standalone functions for convenience
def create_collection(
    collection_name: str,
    embeddings,
    use_cloud: bool = False,
    overwrite: bool = False,
    enable_hybrid: bool = False
) -> QdrantClient:
    """
    Create a Qdrant collection.

    Args:
        collection_name: Name of the collection
        embeddings: Embedding model instance
        use_cloud: Use cloud Qdrant
        overwrite: Delete existing and create new
        enable_hybrid: Enable BM25 for hybrid search

    Returns:
        QdrantClient instance
    """
    manager = CollectionManager(embeddings)
    return manager.create_collection(collection_name, use_cloud, overwrite, enable_hybrid)


def create_hybrid_collection(
    source_collection: str = "recursive_character",
    target_collection: str = "hybrid",
    embeddings=None,
    use_cloud: bool = True,
    overwrite: bool = False
):
    """
    Create hybrid collection from existing collection.

    Args:
        source_collection: Source collection name
        target_collection: Target hybrid collection name
        embeddings: Embedding model (optional, only needed for manager init)
        use_cloud: Use cloud Qdrant
        overwrite: Recreate if exists
    """
    if embeddings is None:
        from retrieval_playground.utils.model_manager import model_manager
        embeddings = model_manager.get_embeddings()

    manager = CollectionManager(embeddings)
    manager.create_hybrid_from_existing(source_collection, target_collection, use_cloud, overwrite)
