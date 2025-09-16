"""
Pre-retrieval chunking strategies for document processing.
"""

from pathlib import Path
from typing import List
from enum import Enum

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader
from docling.document_converter import DocumentConverter
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

from retrieval_playground.utils.model_manager import model_manager
from retrieval_playground.utils.pylogger import get_python_logger
from retrieval_playground.utils import constants, config
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from docling.chunking import HybridChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_docling.loader import ExportType
from langchain_docling import DoclingLoader
import uuid
import time

import warnings
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    BASELINE = "baseline"
    RECURSIVE_CHARACTER = "recursive_character"
    UNSTRUCTURED = "unstructured"
    DOCLING = "docling"
    SEMANTIC = "semantic"


class PreRetrievalChunking:
    """Pre-retrieval chunking strategies for document processing."""

    def __init__(self):
        """Initialize the chunking strategies."""
        self.logger = get_python_logger(log_level=constants.PYTHON_LOG_LEVEL)
        self.embeddings = model_manager.get_embeddings()

        # Initialize chunking strategies
        self._init_chunking_strategies()

        self.logger.info("✅ PreRetrievalChunking initialized")

    def _init_chunking_strategies(self) -> None:
        """Initialize all chunking strategies."""
        # Recursive Character Text Splitter
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Semantic Chunker
        self.semantic_chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile"
        )

        # Document converter for Docling
        self.doc_converter = DocumentConverter()

        self.logger.info("✅ All chunking strategies initialized")

    def load_and_chunk_documents(
        self,
        pdf_directory: str,
        strategy: ChunkingStrategy,
        vector_store: QdrantVectorStore
    ) -> List[Document]:
        """
        Load and chunk documents using the specified strategy.

        Args:
            pdf_directory: Path to directory containing PDF files
            strategy: Chunking strategy to use
            vector_store: QdrantVectorStore to add chunks to
        Returns:
            List of chunked Document objects
        """
        start_time = time.time()
        self.logger.info(f"🔄 Loading and chunking documents with strategy: {strategy.value}")

        try:
            if strategy == ChunkingStrategy.BASELINE:
                result = self._baseline_chunking(pdf_directory, vector_store)
            elif strategy == ChunkingStrategy.RECURSIVE_CHARACTER:
                result = self._recursive_character_chunking(pdf_directory, vector_store)
            elif strategy == ChunkingStrategy.SEMANTIC:
                result = self._semantic_chunking(pdf_directory, vector_store)
            elif strategy == ChunkingStrategy.DOCLING:
                result = self._docling_chunking(pdf_directory, vector_store)
            elif strategy == ChunkingStrategy.UNSTRUCTURED:
                result = self._unstructured_chunking(pdf_directory, vector_store)
            else:
                raise ValueError(f"Unknown chunking strategy: {strategy}")

            elapsed_time = time.time() - start_time
            self.logger.info(f"⏱️ Strategy {strategy.value} completed in {elapsed_time:.2f} seconds")
            return result

        except Exception as e:
            self.logger.error(f"❌ Failed to process documents with {strategy.value}: {e}")
            raise

    def _baseline_chunking(self, pdf_directory: str, vector_store: QdrantVectorStore) -> List[Document]:
        """Baseline chunking using CharacterTextSplitter."""
        from langchain.text_splitter import CharacterTextSplitter

        pdf_path = Path(pdf_directory)
        total_chunks = 0

        # Load PDFs
        for pdf_file in pdf_path.glob("*.pdf"):
            try:
                self.logger.info(f"📄 Processing {pdf_file.name} with baseline chunking")
                loader = PyPDFLoader(str(pdf_file))
                pdf_docs = loader.load()

                # Add source metadata
                for doc in pdf_docs:
                    doc.metadata["source"] = pdf_file.name
                    doc.metadata["chunking_strategy"] = ChunkingStrategy.BASELINE.value
                # Chunk documents
                splitter = CharacterTextSplitter(
                    chunk_size=3000,
                    chunk_overlap=200,
                    separator="\n"
                )
                chunked_docs = splitter.split_documents(pdf_docs)
                # Add chunk metadata
                for i, chunk in enumerate(chunked_docs):
                    chunk.metadata["chunk_id"] = uuid.uuid4()

                vector_store.add_documents(chunked_docs)
                total_chunks += len(chunked_docs)
                self.logger.info(f"✅ Added {len(chunked_docs)} chunks to baseline vector store for {pdf_file.name}")


            except Exception as e:
                self.logger.error(f"❌ Failed to load {pdf_file.name}: {e}")
                continue

        self.logger.info(f"✅ Baseline chunking completed: {total_chunks} total chunks")

    def _recursive_character_chunking(self, pdf_directory: str, vector_store: QdrantVectorStore) -> List[Document]:
        """Recursive character text splitting."""
        pdf_path = Path(pdf_directory)
        total_chunks = 0

        # Load PDFs
        for pdf_file in pdf_path.glob("*.pdf"):
            try:
                self.logger.info(f"📄 Processing {pdf_file.name} with recursive character chunking")
                loader = PyPDFLoader(str(pdf_file))
                pdf_docs = loader.load()

                # Add source metadata
                for doc in pdf_docs:
                    doc.metadata["source"] = pdf_file.name
                    doc.metadata["chunking_strategy"] = ChunkingStrategy.RECURSIVE_CHARACTER.value

                # Chunk documents
                chunked_docs = self.recursive_splitter.split_documents(pdf_docs)

                # Add chunk metadata
                for i, chunk in enumerate(chunked_docs):
                    chunk.metadata["chunk_id"] = uuid.uuid4()

                vector_store.add_documents(chunked_docs)
                total_chunks += len(chunked_docs)
                self.logger.info(f"✅ Added {len(chunked_docs)} chunks to recursive character vector store for {pdf_file.name}")

            except Exception as e:
                self.logger.error(f"❌ Failed to load {pdf_file.name}: {e}")
                continue

        self.logger.info(f"✅ Recursive character chunking completed: {total_chunks} total chunks")


    def _semantic_chunking(self, pdf_directory: str, vector_store: QdrantVectorStore) -> List[Document]:
        """Semantic chunking using embeddings."""
        pdf_path = Path(pdf_directory)
        total_chunks = 0

        # Load PDFs
        for pdf_file in pdf_path.glob("*.pdf"):
            try:
                self.logger.info(f"📄 Processing {pdf_file.name} with semantic chunking")
                loader = PyPDFLoader(str(pdf_file))
                pdf_docs = loader.load()

                # Add source metadata
                for doc in pdf_docs:
                    doc.metadata["source"] = pdf_file.name
                    doc.metadata["chunking_strategy"] = ChunkingStrategy.SEMANTIC.value

                # Chunk documents semantically
                chunked_docs = self.semantic_chunker.split_documents(pdf_docs)

                # Add chunk metadata
                for i, chunk in enumerate(chunked_docs):
                    chunk.metadata["chunk_id"] = uuid.uuid4()
                total_chunks += len(chunked_docs)

                vector_store.add_documents(chunked_docs)
                self.logger.info(f"✅ Added {len(chunked_docs)} chunks to semantic vector store for {pdf_file.name}")

            except Exception as e:
                self.logger.error(f"❌ Failed to load {pdf_file.name}: {e}")
                continue

        self.logger.info(f"✅ Semantic chunking completed: {total_chunks} total chunks")


    def _docling_chunking(self, pdf_directory: str, vector_store: QdrantVectorStore) -> List[Document]:
        """Docling chunking for PDF processing."""
        pdf_path = Path(pdf_directory)
        total_chunks = 0
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
            ],
            strip_headers=False
        )

        for pdf_file in pdf_path.glob("*.pdf"):
            try:
                self.logger.info(f"📄 Processing {pdf_file.name} with Docling chunking")

                # Use Docling to convert PDF
                loader = DoclingLoader(
                    file_path=pdf_file,
                    export_type=ExportType.MARKDOWN,
                    chunker=HybridChunker(tokenizer=constants.DEFAULT_EMBEDDING_MODEL, max_tokens=1000))

                docs = loader.load()
                splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]

                # Create document
                chunked_docs = [Document(
                    page_content=split.page_content,
                    metadata={
                        "source": pdf_file.name,
                        "chunking_strategy": ChunkingStrategy.DOCLING.value
                    }
                ) for split in splits]

                # Add chunk metadata
                for i, chunk in enumerate(chunked_docs):
                    chunk.metadata["chunk_id"] = i

                vector_store.add_documents(chunked_docs)
                total_chunks += len(chunked_docs)
                self.logger.info(f"✅ Added {len(chunked_docs)} chunks to docling vector store for {pdf_file.name}")

            except Exception as e:
                self.logger.error(f"❌ Failed to process {pdf_file.name} with Docling: {e}")
                continue

        self.logger.info(f"✅ Docling chunking completed: {total_chunks} total chunks")


    def _unstructured_chunking(self, pdf_directory: str, vector_store: QdrantVectorStore) -> List[Document]:
        """Unstructured library chunking with chunk_by_title."""
        pdf_path = Path(pdf_directory)
        total_chunks = 0

        for pdf_file in pdf_path.glob("*.pdf"):
            try:
                self.logger.info(f"📄 Processing {pdf_file.name} with Unstructured chunk_by_title")

                # Partition PDF into elements
                elements = partition_pdf(
                    filename=str(pdf_file),
                    strategy="fast",
                    infer_table_structure=True
                )

                # Chunk by title for better structure awareness
                chunked_elements = chunk_by_title(
                    elements,
                    max_characters=3000,
                    combine_text_under_n_chars=3000,
                    new_after_n_chars=800
                )
                documents = []
                # Convert elements to LangChain Documents
                for element in chunked_elements:
                    doc = Document(
                        page_content=str(element),
                        metadata={
                            "source": pdf_file.name,
                            "chunking_strategy": ChunkingStrategy.UNSTRUCTURED.value,
                        }
                    )
                    documents.append(doc)

                # Add chunk metadata
                for i, chunk in enumerate(documents):
                    chunk.metadata["chunk_id"] = uuid.uuid4()

                vector_store.add_documents(documents)
                total_chunks += len(documents)
                self.logger.info(f"✅ Added {len(documents)} chunks to unstructured vector store for {pdf_file.name}")

            except Exception as e:
                self.logger.error(f"❌ Failed to process {pdf_file.name} with Unstructured: {e}")
                continue

        self.logger.info(f"✅ Unstructured chunk_by_title completed: {total_chunks} total chunks")


    def get_available_strategies(self) -> List[ChunkingStrategy]:
        """Get list of available chunking strategies."""
        return list(ChunkingStrategy)

    def create_and_store_chunks(self, pdf_directory: str, strategy: ChunkingStrategy, use_cloud: bool = False) -> List[Document]:
        """Create and store chunks for a given strategy."""
        self.logger.info(f"🏗️ Creating and storing chunks for strategy: {strategy.value}")

        if use_cloud:
            # Cloud Qdrant setup
            self.logger.info(f"🔧 Initializing cloud Qdrant client and collection: {strategy.value}")
            qdrant_client = QdrantClient(
                url=constants.QDRANT_URL,
                api_key=constants.QDRANT_KEY
            )

            # Check if collection already exists
            collections = qdrant_client.get_collections()
            collection_exists = any(col.name == strategy.value for col in collections.collections)

            if collection_exists:
                self.logger.info(f"✅ Collection {strategy.value} already exists, skipping chunk creation")
                return
        else:
            # Local Qdrant setup (original behavior)
            qdrant_path = config.QDRANT_DIR / strategy.value
            exists = qdrant_path.exists() and any(qdrant_path.iterdir())
            if exists:
                self.logger.info(f"✅ Collection {strategy.value} already exists, skipping chunk creation")
                return

            self.logger.info(f"🔧 Initializing local Qdrant client and collection: {strategy.value}")
            qdrant_client = QdrantClient(path=str(qdrant_path))

        qdrant_client.create_collection(
            collection_name=strategy.value,
            vectors_config=VectorParams(
                size=len(self.embeddings.embed_query('size')),
                distance=Distance.COSINE
            )
        )

        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=strategy.value,
            embedding=self.embeddings,
        )

        self.load_and_chunk_documents(pdf_directory, strategy, vector_store)

        self.logger.info(f"🔒 Cleaning up resources for {strategy.value}")
        qdrant_client.close()
        del qdrant_client
        del vector_store

        self.logger.info(f"✅ Completed chunk creation and storage for {strategy.value}")

    def create_chunks_for_all_strategies(self, pdf_directory: str, use_cloud: bool = False) -> None:
        """Create chunks for all strategies."""
        strategies = self.get_available_strategies()
        storage_type = "cloud" if use_cloud else "local"
        self.logger.info(f"🚀 Starting chunk creation for {len(strategies)} strategies using {storage_type} Qdrant")

        for strategy in strategies:
            self.create_and_store_chunks(pdf_directory, strategy, use_cloud)

        self.logger.info(f"🎉 Completed chunk creation for all {len(strategies)} strategies using {storage_type} Qdrant")


if __name__ == "__main__":
    chunking = PreRetrievalChunking()
    chunking.create_chunks_for_all_strategies(pdf_directory=str(config.SAMPLE_PAPERS_DIR), use_cloud=False)