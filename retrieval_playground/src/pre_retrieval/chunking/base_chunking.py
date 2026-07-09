"""
Base class for all chunking strategies.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import uuid

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from retrieval_playground.utils.pylogger import get_python_logger
from retrieval_playground.utils import config


class BaseChunking(ABC):
    """
    Base class for chunking strategies.

    All chunking strategies inherit from this class and implement
    the chunk_documents() method.
    """

    def __init__(self, strategy_name: str):
        """
        Initialize base chunking strategy.

        Args:
            strategy_name: Name of the chunking strategy
        """
        self.strategy_name = strategy_name
        self.logger = get_python_logger(log_level=config.PYTHON_LOG_LEVEL)

    @abstractmethod
    def chunk_pdf_directory(
        self,
        pdf_directory: str
    ) -> List[Document]:
        """
        Chunk all PDFs in a directory.

        Each strategy implements this differently.

        Args:
            pdf_directory: Path to directory containing PDF files

        Returns:
            List of chunked Document objects
        """
        pass

    @abstractmethod
    def chunk_single_pdf(self, pdf_path: str) -> List[Document]:
        """
        Chunk a single PDF file without storing to vector database.

        Useful for demos, testing, and notebooks.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of chunked Document objects

        Example:
            chunks = strategy.chunk_single_pdf("path/to/file.pdf")
            print(f"Created {len(chunks)} chunks")
        """
        pass

    def load_pdf(self, pdf_file: Path) -> List[Document]:
        """
        Load a PDF file.

        Args:
            pdf_file: Path to PDF file

        Returns:
            List of Document objects
        """
        loader = PyPDFLoader(str(pdf_file))
        return loader.load()

    def add_metadata(self, docs: List[Document], pdf_file: Path) -> None:
        """
        Add metadata to documents.

        Args:
            docs: List of documents
            pdf_file: Source PDF file
        """
        for doc in docs:
            doc.metadata["source"] = pdf_file.name
            doc.metadata["chunking_strategy"] = self.strategy_name

    def add_chunk_ids(self, chunks: List[Document]) -> None:
        """
        Add unique IDs to chunks.

        Args:
            chunks: List of chunk documents
        """
        for chunk in chunks:
            if "chunk_id" not in chunk.metadata:
                chunk.metadata["chunk_id"] = str(uuid.uuid4())

    def process_pdf_directory(
        self,
        pdf_directory: str,
        process_fn
    ) -> int:
        """
        Process all PDFs in a directory.

        Args:
            pdf_directory: Path to directory containing PDFs
            process_fn: Function to process each PDF

        Returns:
            Total number of chunks created
        """
        pdf_path = Path(pdf_directory)
        total_chunks = 0

        for pdf_file in pdf_path.glob("*.pdf"):
            try:
                self.logger.info(f"  Processing: {pdf_file.name}")
                chunks_count = process_fn(pdf_file)
                total_chunks += chunks_count
                self.logger.info(f"  ✅ Created {chunks_count} chunks")

            except Exception as e:
                self.logger.error(f"  ❌ Failed to process {pdf_file.name}: {e}")
                continue

        return total_chunks
