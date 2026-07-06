"""
Recursive Character Chunking Strategy.

WHAT: Simple baseline that respects document boundaries
HOW: Splits at paragraphs → sentences → words (in that order)
BEST FOR: Most documents, learning RAG, fast iteration
"""

from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from .base_chunking import BaseChunking


class RecursiveChunking(BaseChunking):
    """
    Recursive Character Chunking (Simple Baseline).

    Splits text at natural boundaries:
    1. Try paragraphs (\\n\\n)
    2. Try sentences (\\n)
    3. Try words (space)
    4. Last resort: characters

    Parameters:
        - chunk_size: 512 tokens (2026 best practice)
        - chunk_overlap: 100 tokens (~20%)
    """

    def __init__(self):
        """Initialize Recursive Character Chunking."""
        super().__init__(strategy_name="recursive_character")

        # Create splitter (respects boundaries)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,             # 512 tokens (optimal for retrieval)
            chunk_overlap=205,           # 10% overlap
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Try in order
        )

        self.logger.info("✅ Recursive Character Chunking initialized")

    def chunk_single_pdf(self, pdf_path: str) -> List[Document]:
        """Chunk a single PDF without storing to vector database."""
        pdf_file = Path(pdf_path)
        pdf_docs = self.load_pdf(pdf_file)
        self.add_metadata(pdf_docs, pdf_file)
        chunks = self.splitter.split_documents(pdf_docs)
        self.add_chunk_ids(chunks)
        return chunks

    def chunk_documents(
        self,
        pdf_directory: str,
        vector_store: QdrantVectorStore
    ) -> None:
        """
        Chunk documents using recursive character splitting.

        Memory Management:
        - Processes one PDF at a time
        - Pushes chunks to Qdrant immediately after each file
        - Clears memory before next file

        Args:
            pdf_directory: Path to directory containing PDF files
            vector_store: QdrantVectorStore to store chunks
        """
        self.logger.info("📝 Starting Recursive Character Chunking")

        def process_pdf(pdf_file: Path) -> int:
            # Step 1: Load PDF
            pdf_docs = self.load_pdf(pdf_file)

            # Step 2: Add metadata
            self.add_metadata(pdf_docs, pdf_file)

            # Step 3: Split into chunks (respects boundaries)
            chunks = self.splitter.split_documents(pdf_docs)

            # Step 4: Add unique IDs
            self.add_chunk_ids(chunks)

            # Step 5: Push to Qdrant immediately (one file at a time)
            vector_store.add_documents(chunks)
            self.logger.info(f"    ✓ Pushed {len(chunks)} chunks to Qdrant")

            # Step 6: Clear memory
            chunk_count = len(chunks)
            del pdf_docs
            del chunks

            return chunk_count

        # Process all PDFs (one at a time for memory efficiency)
        total = self.process_pdf_directory(pdf_directory, process_pdf)

        self.logger.info(f"🎉 Recursive chunking complete: {total} total chunks")
