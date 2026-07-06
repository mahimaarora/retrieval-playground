"""
Hybrid Chunking: Creates chunks with BOTH dense + sparse vectors

Why?
- Dense vectors: Semantic similarity (embeddings)
- Sparse vectors: Keyword matching (BM25-style)
- Both uploaded to Qdrant cloud together

Simple to use:
    from hybrid_chunking import HybridChunking

    chunker = HybridChunking()
    chunker.chunk_documents("data/pdfs", vector_store)
"""

from pathlib import Path
from typing import List, Dict, Any
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client.models import PointStruct, VectorParams, Distance, SparseVectorParams, SparseIndexParams

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from retrieval_playground.utils.pylogger import get_python_logger
from retrieval_playground.utils import constants
from retrieval_playground.src.pre_retrieval.chunking.base_chunking import BaseChunking


class HybridChunking(BaseChunking):
    """
    Hybrid chunking with dense + sparse vectors.

    How it works:
    1. Chunk documents normally (recursive splitting)
    2. Create dense embeddings (semantic)
    3. Create sparse vectors (BM25-style keywords)
    4. Upload both to Qdrant cloud

    Benefits:
    - Enables true cloud-based hybrid search
    - No local BM25 index needed
    - Scales to millions of documents
    """

    def __init__(self):
        """Initialize hybrid chunking."""
        super().__init__(strategy_name="hybrid")

        # Text splitter (same as recursive chunking)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=205,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # TF-IDF vectorizer for sparse vectors
        # Simple word-level, limited vocabulary for efficiency
        self.vectorizer = TfidfVectorizer(
            max_features=1000,      # Top 1000 terms
            lowercase=True,
            token_pattern=r'\b\w+\b',  # Word tokens
            min_df=1                # Keep rare terms (important for BM25)
        )

        self.logger.info("✅ Hybrid Chunking initialized (dense + sparse)")

    def chunk_single_pdf(self, pdf_path: str) -> List[Document]:
        """
        Chunk a single PDF without storing to vector database.

        Note: This method only creates the text chunks with basic metadata.
        Sparse vectors are NOT generated here (they require a corpus-wide vocabulary).
        Use chunk_documents() for full hybrid chunking with dense + sparse vectors.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of chunked Document objects with basic metadata
        """
        pdf_file = Path(pdf_path)

        # Load PDF
        pdf_docs = self.load_pdf(pdf_file)

        # Add metadata
        self.add_metadata(pdf_docs, pdf_file)

        # Chunk documents
        chunks = self.splitter.split_documents(pdf_docs)

        # Add chunk IDs
        self.add_chunk_ids(chunks)

        return chunks

    def _create_sparse_vector(self, text: str, vocab: Dict[str, int]) -> Dict[str, List]:
        """
        Create sparse vector from text using TF-IDF.

        Args:
            text: Document text
            vocab: Vocabulary mapping from vectorizer

        Returns:
            Sparse vector dict with indices and values
        """
        # Tokenize and count terms
        words = text.lower().split()
        term_counts = {}

        for word in words:
            if word in vocab:
                idx = vocab[word]
                term_counts[idx] = term_counts.get(idx, 0) + 1

        if not term_counts:
            # Empty sparse vector
            return {"indices": [], "values": []}

        # Convert to sparse format
        indices = list(term_counts.keys())
        values = list(term_counts.values())

        return {"indices": indices, "values": values}

    def chunk_documents(
        self,
        pdf_directory: str,
        vector_store: QdrantVectorStore
    ) -> None:
        """
        Chunk documents and upload with dense + sparse vectors.

        Args:
            pdf_directory: Path to PDF files
            vector_store: QdrantVectorStore to upload to
        """
        from langchain_community.document_loaders import PyPDFLoader

        self.logger.info("📝 Starting Hybrid Chunking (dense + sparse)")

        pdf_dir = Path(pdf_directory)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))

        if not pdf_files:
            self.logger.warning(f"No PDF files found in {pdf_directory}")
            return

        # Step 1: Load all documents first to build vocabulary
        self.logger.info("🔍 Step 1/3: Loading documents and building vocabulary...")
        all_texts = []
        all_chunks = []

        for pdf_file in pdf_files:
            self.logger.info(f"  Processing: {pdf_file.name}")

            # Load PDF
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()

            # Chunk
            chunks = self.splitter.split_documents(pages)

            # Collect texts for vocabulary building
            for chunk in chunks:
                all_texts.append(chunk.page_content)
                all_chunks.append(chunk)

        # Step 2: Build TF-IDF vocabulary from all documents
        self.logger.info(f"Step 2/3: Building vocabulary from {len(all_texts)} chunks...")
        self.vectorizer.fit(all_texts)
        vocab = self.vectorizer.vocabulary_
        self.logger.info(f"  Vocabulary size: {len(vocab)} terms")

        # Step 3: Create dense + sparse vectors and upload
        self.logger.info(f"Step 3/3: Creating vectors and uploading to Qdrant...")

        points = []
        for i, chunk in enumerate(all_chunks):
            # Create dense embedding (semantic)
            dense_vector = vector_store.embeddings.embed_query(chunk.page_content)

            # Create sparse vector (keywords)
            sparse_vector = self._create_sparse_vector(chunk.page_content, vocab)

            # Create point with both vectors
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vector,
                    "sparse": sparse_vector
                },
                payload={
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                }
            )

            points.append(point)

            # Upload in batches of 100
            if len(points) >= 100:
                vector_store.client.upsert(
                    collection_name=vector_store.collection_name,
                    points=points
                )
                self.logger.info(f"  ✓ Uploaded {i+1}/{len(all_chunks)} chunks")
                points = []

        # Upload remaining points
        if points:
            vector_store.client.upsert(
                collection_name=vector_store.collection_name,
                points=points
            )

        self.logger.info(f"✅ Hybrid chunking complete: {len(all_chunks)} total chunks")
        self.logger.info(f"   - Dense vectors: {len(all_chunks)} (semantic)")
        self.logger.info(f"   - Sparse vectors: {len(all_chunks)} (keywords)")


def create_hybrid_collection(
    collection_name: str = "hybrid",
    embedding_size: int = 896,  # Qwen3-Embedding size
    use_cloud: bool = True
):
    """
    Helper function to create a Qdrant collection with dense + sparse vectors.

    Args:
        collection_name: Name for the collection
        embedding_size: Size of dense embeddings
        use_cloud: Use cloud Qdrant (True) or local (False)

    Returns:
        QdrantClient instance

    Example:
        # Create collection
        client = create_hybrid_collection("hybrid", use_cloud=True)

        # Use for chunking
        chunker = HybridChunking()
        vector_store = QdrantVectorStore(client=client, collection_name="hybrid", ...)
        chunker.chunk_documents("data/pdfs", vector_store)
    """
    from qdrant_client import QdrantClient
    from retrieval_playground.utils import config

    # Setup client
    if use_cloud:
        client = QdrantClient(
            url=constants.QDRANT_URL,
            api_key=constants.QDRANT_KEY
        )
        print(f"☁️  Using cloud Qdrant")
    else:
        qdrant_path = config.QDRANT_DIR / collection_name
        client = QdrantClient(path=str(qdrant_path))
        print(f"💾 Using local Qdrant at {qdrant_path}")

    # Check if collection exists
    collections = client.get_collections()
    if any(col.name == collection_name for col in collections.collections):
        print(f"✅ Collection '{collection_name}' already exists")
        return client

    # Create collection with dense + sparse vectors
    print(f"📦 Creating collection '{collection_name}' with dense + sparse vectors...")

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=embedding_size,
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=False  # Keep in memory for speed
                )
            )
        }
    )

    print(f"✅ Collection created successfully!")
    print(f"   - Dense vector: {embedding_size} dimensions (cosine)")
    print(f"   - Sparse vector: TF-IDF style (keyword matching)")

    return client


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    """
    Example: Create hybrid collection and chunk documents.

    Run with:
        python -m retrieval_playground.src.pre_retrieval.chunking.hybrid_chunking
    """
    from langchain_qdrant import QdrantVectorStore
    from retrieval_playground.utils.model_manager import model_manager
    from retrieval_playground.utils import config

    # Step 1: Create hybrid collection
    print("=" * 70)
    print("STEP 1: Create Hybrid Collection")
    print("=" * 70)

    client = create_hybrid_collection(
        collection_name="hybrid",
        embedding_size=896,  # Qwen3-Embedding size
        use_cloud=True
    )

    # Step 2: Setup vector store
    print("\n" + "=" * 70)
    print("STEP 2: Setup Vector Store")
    print("=" * 70)

    embeddings = model_manager.get_embeddings()
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="hybrid",
        embedding=embeddings
    )
    print("✅ Vector store ready")

    # Step 3: Chunk documents
    print("\n" + "=" * 70)
    print("STEP 3: Chunk Documents")
    print("=" * 70)

    chunker = HybridChunking()
    chunker.chunk_documents(
        pdf_directory=str(config.WORKSHOP_DATA_DIR),
        vector_store=vector_store
    )

    print("\n" + "=" * 70)
    print("✅ COMPLETE: Hybrid collection ready for cloud-based hybrid search!")
    print("=" * 70)

    client.close()
