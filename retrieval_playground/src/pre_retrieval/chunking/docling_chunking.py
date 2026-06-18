"""
Docling Multimodal Chunking Strategy.

WHAT: Structure-aware + multimodal chunking (text + tables + images)
HOW: Uses Docling to extract and parse all content types from PDFs
BEST FOR: Research papers, technical docs, complex PDFs with figures/tables
"""

from pathlib import Path
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from .base_chunking import BaseChunking
from .docling_multimodal import DoclingMultimodalParser, TextChunk, TableChunk, ImageChunk


class DoclingChunking(BaseChunking):
    """
    Docling Multimodal Chunking (Text + Tables + Images).

    How it works:
    1. Uses Docling DocumentConverter to parse PDFs
    2. Extracts text chunks using HybridChunker (smart segmentation)
    3. Extracts tables with structure preservation
    4. Extracts images and saves to disk
    5. Generates AI descriptions for tables and images (using Gemini)
    6. Stores all chunks in Qdrant with metadata

    Chunk Types:
        - TEXT: Regular text content (1000 tokens max)
        - TABLE: Structured data with AI-generated description
        - IMAGE: Images with AI-generated description and saved file path

    Features:
        - Structure-aware text chunking
        - Accurate table extraction with cell matching
        - Image extraction with base64 encoding
        - AI-powered descriptions for better retrieval
        - Preserves document hierarchy (headings, sections)

    Trade-offs:
        - Slower than text-only chunking (AI descriptions)
        - Requires GOOGLE_API_KEY for AI descriptions
        - Saves images to disk (data/images/)
    """

    def __init__(self):
        """Initialize Docling Multimodal Chunking."""
        super().__init__(strategy_name="docling")

        # Initialize multimodal parser
        self.parser = DoclingMultimodalParser(
            images_scale=2.0,           # 144 DPI for images
            table_mode="accurate",      # Accurate table extraction
            do_cell_matching=True,      # Map table structure to PDF cells
            generate_descriptions=True, # AI descriptions for images/tables
            images_output_dir="data/images"  # Save images here
        )

        self.logger.info("✅ Docling Multimodal Chunking initialized")

    def chunk_documents(
        self,
        pdf_directory: str,
        vector_store: QdrantVectorStore
    ) -> None:
        """
        Chunk documents using Docling multimodal extraction.

        Memory Management:
        - Processes one PDF at a time
        - Pushes chunks to Qdrant immediately after each file
        - Clears memory before next file
        - Images saved to disk (not kept in memory)

        Args:
            pdf_directory: Path to directory containing PDF files
            vector_store: QdrantVectorStore to store chunks
        """
        self.logger.info("🎨 Starting Docling Multimodal Chunking")
        self.logger.info("    📝 Extracting: Text + Tables + Images")

        total_text_chunks = 0
        total_table_chunks = 0
        total_image_chunks = 0

        def process_pdf(pdf_file: Path) -> int:
            nonlocal total_text_chunks, total_table_chunks, total_image_chunks

            # Step 1: Parse PDF to extract all chunk types
            chunks = self.parser.parse(str(pdf_file))

            if not chunks:
                self.logger.warning(f"    ⚠️  No chunks extracted from {pdf_file.name}")
                return 0

            # Step 2: Convert to LangChain Documents for Qdrant
            documents = []

            for chunk in chunks:
                # Prepare metadata
                metadata = {
                    "source": pdf_file.name,
                    "chunking_strategy": self.strategy_name,
                    "chunk_type": chunk.chunk_type,  # "text", "table", or "image"
                    "chunk_id": chunk.chunk_id,
                    "sequence_number": chunk.sequence_number,
                }

                # Add optional metadata
                if chunk.source_page:
                    metadata["source_page"] = chunk.source_page
                if chunk.parent_heading:
                    metadata["parent_heading"] = chunk.parent_heading

                # Add type-specific metadata
                if isinstance(chunk, TextChunk):
                    metadata["word_count"] = chunk.word_count
                    metadata["char_count"] = chunk.char_count
                    total_text_chunks += 1

                elif isinstance(chunk, TableChunk):
                    metadata["num_rows"] = chunk.num_rows
                    metadata["num_cols"] = chunk.num_cols
                    metadata["columns"] = chunk.get_columns()
                    # Store table data as JSON
                    metadata["table_data"] = chunk.dataframe.to_dict(orient="records")
                    total_table_chunks += 1

                elif isinstance(chunk, ImageChunk):
                    if chunk.image_path:
                        metadata["image_path"] = chunk.image_path
                    if chunk.image_format:
                        metadata["image_format"] = chunk.image_format
                    if chunk.image_type:
                        metadata["image_type"] = chunk.image_type
                    # Note: We don't store base64 in Qdrant (too large)
                    # Image files are saved to disk, retrievable via image_path
                    total_image_chunks += 1

                # Create LangChain Document
                doc = Document(
                    page_content=chunk.content,  # Text or AI description
                    metadata=metadata
                )
                documents.append(doc)

            # Step 3: Push to Qdrant immediately (one file at a time)
            vector_store.add_documents(documents)

            text_count = sum(1 for c in chunks if isinstance(c, TextChunk))
            table_count = sum(1 for c in chunks if isinstance(c, TableChunk))
            image_count = sum(1 for c in chunks if isinstance(c, ImageChunk))

            self.logger.info(
                f"    ✓ Pushed {len(documents)} chunks to Qdrant "
                f"({text_count} text, {table_count} tables, {image_count} images)"
            )

            # Step 4: Clear memory
            chunk_count = len(chunks)
            del chunks
            del documents

            return chunk_count

        # Process all PDFs (one at a time for memory efficiency)
        total = self.process_pdf_directory(pdf_directory, process_pdf)

        self.logger.info(
            f"🎉 Docling multimodal chunking complete: {total} total chunks\n"
            f"   - {total_text_chunks} text chunks\n"
            f"   - {total_table_chunks} table chunks\n"
            f"   - {total_image_chunks} image chunks"
        )
