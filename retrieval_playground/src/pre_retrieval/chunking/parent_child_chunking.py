"""
Parent-Child Chunking Strategy.

WHAT: Small chunks for search, large chunks for context
HOW: Creates child chunks (retrieval) linked to parent chunks (generation)
BEST FOR: Production RAG systems, complex Q&A
"""

from pathlib import Path
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from .base_chunking import BaseChunking


class ParentChildChunking(BaseChunking):
    """
    Parent-Child Chunking (Production Pattern).

    How it works:
    1. Create LARGE parent chunks (2000 tokens) → stored separately
    2. Create SMALL child chunks (400 tokens) → stored in vector DB
    3. Search: Use child chunks (precise retrieval)
    4. Generate: Return parent chunks (full context)

    Example:
        Parent (2000 tokens): [Full chapter on ML]
          ├─ Child 1 (400 tokens): [ML Introduction]
          ├─ Child 2 (400 tokens): [Supervised Learning]
          └─ Child 3 (400 tokens): [Examples]

    Query: "Explain supervised learning"
    → Finds: Child 2 (precise match)
    → Returns: Parent (full chapter with context)
    """

    def __init__(self):
        """Initialize Parent-Child Chunking."""
        super().__init__(strategy_name="parent_child")

        # Large parent chunks (for context)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=6144,             # 1536 tokens (large context)
            chunk_overlap=614,           # 10% overlap
            separators=["\n\n", "\n", " ", ""]
        )

        # Small child chunks (for retrieval)
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1536,             # 384 tokens (precise retrieval)
            chunk_overlap=154,           # 10% overlap
            separators=["\n\n", "\n", " ", ""]
        )

        self.logger.info("✅ Parent-Child Chunking initialized")

    def chunk_single_pdf(self, pdf_path: str) -> List[Document]:
        """Chunk a single PDF into parent-child structure."""
        pdf_file = Path(pdf_path)
        pdf_docs = self.load_pdf(pdf_file)

        # Create parent chunks
        parent_chunks = self.parent_splitter.split_documents(pdf_docs)

        all_chunks = []
        for parent_idx, parent_doc in enumerate(parent_chunks):
            # Generate parent ID (consistent with chunk_documents)
            parent_id = f"{pdf_file.stem}_parent_{parent_idx}"

            # Create parent document with full metadata
            parent_document = Document(
                page_content=parent_doc.page_content,
                metadata={
                    "source": pdf_file.name,
                    "chunking_strategy": self.strategy_name,
                    "chunk_type": "parent",
                    "parent_id": parent_id,
                    "parent_index": parent_idx
                }
            )
            all_chunks.append(parent_document)

            # Create child chunks from this parent
            child_chunks = self.child_splitter.split_documents([parent_doc])

            for child_chunk in child_chunks:
                child_chunk.metadata["source"] = pdf_file.name
                child_chunk.metadata["chunking_strategy"] = self.strategy_name
                child_chunk.metadata["chunk_type"] = "child"
                child_chunk.metadata["parent_id"] = parent_id
                child_chunk.metadata["parent_chunk_index"] = parent_idx

            self.add_chunk_ids(child_chunks)
            all_chunks.extend(child_chunks)

        # Add chunk IDs to parent chunks
        self.add_chunk_ids([chunk for chunk in all_chunks if chunk.metadata.get("chunk_type") == "parent"])

        return all_chunks

    def chunk_documents(
        self,
        pdf_directory: str,
        vector_store: QdrantVectorStore
    ) -> None:
        """
        Chunk documents using parent-child strategy.

        Memory Management:
        - Processes one PDF at a time
        - Pushes child chunks to Qdrant immediately after each file
        - Stores parent chunks in memory (needed for retrieval)
        - Clears temporary data before next file

        Args:
            pdf_directory: Path to directory containing PDF files
            vector_store: QdrantVectorStore to store child chunks
        """
        self.logger.info("Starting Parent-Child Chunking")

        total_parent_chunks = 0
        total_child_chunks = 0

        def process_pdf(pdf_file: Path) -> int:
            nonlocal total_parent_chunks

            # Step 1: Load PDF
            pdf_docs = self.load_pdf(pdf_file)

            # Step 2: Add metadata
            self.add_metadata(pdf_docs, pdf_file)

            # Step 3: Create parent chunks (large)
            parent_chunks = self.parent_splitter.split_documents(pdf_docs)

            # Step 4: Process each parent and create children
            file_child_count = 0
            all_child_chunks = []
            all_parent_chunks = []

            for parent_idx, parent_doc in enumerate(parent_chunks):
                # Generate unique parent ID
                parent_id = f"{pdf_file.stem}_parent_{parent_idx}"

                # Create parent document with metadata
                parent_document = Document(
                    page_content=parent_doc.page_content,
                    metadata={
                        **parent_doc.metadata,
                        "chunk_type": "parent",  # Mark as parent
                        "parent_id": parent_id,
                        "parent_index": parent_idx
                    }
                )
                all_parent_chunks.append(parent_document)

                # Create child chunks from this parent
                child_chunks = self.child_splitter.split_documents([parent_doc])

                # Add metadata to child chunks
                for child_chunk in child_chunks:
                    child_chunk.metadata["chunk_type"] = "child"  # Mark as child
                    child_chunk.metadata["parent_id"] = parent_id
                    child_chunk.metadata["parent_chunk_index"] = parent_idx

                # Add chunk IDs
                self.add_chunk_ids(child_chunks)

                # Collect all child chunks
                all_child_chunks.extend(child_chunks)
                file_child_count += len(child_chunks)

            # Step 5: Add chunk IDs to parent chunks
            self.add_chunk_ids(all_parent_chunks)

            # Step 6: Push BOTH parent and child chunks to Qdrant
            # Parents first, then children
            vector_store.add_documents(all_parent_chunks)
            vector_store.add_documents(all_child_chunks)

            self.logger.info(
                f"    ✓ Pushed {len(all_parent_chunks)} parent chunks to Qdrant"
            )
            self.logger.info(
                f"    ✓ Pushed {file_child_count} child chunks to Qdrant"
            )

            # Step 7: Clear memory
            total_parent_chunks += len(parent_chunks)
            del pdf_docs
            del parent_chunks
            del all_child_chunks
            del all_parent_chunks

            return file_child_count

        # Process all PDFs (one at a time for memory efficiency)
        total_child_chunks = self.process_pdf_directory(pdf_directory, process_pdf)

        self.logger.info(
            f"✅ Parent-child chunking complete:\n"
            f"   - {total_parent_chunks} parent chunks (stored in memory)\n"
            f"   - {total_child_chunks} child chunks (pushed to Qdrant)"
        )

    def get_parent_chunk(self, vector_store: QdrantVectorStore, parent_id: str) -> Document:
        """
        Get parent chunk by ID from Qdrant.

        Use this during retrieval to get full context.

        Args:
            vector_store: QdrantVectorStore containing the chunks
            parent_id: ID of the parent chunk

        Returns:
            Parent Document with full content
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Search for parent chunk by parent_id
        results = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.parent_id",
                        match=MatchValue(value=parent_id)
                    ),
                    FieldCondition(
                        key="metadata.chunk_type",
                        match=MatchValue(value="parent")
                    )
                ]
            ),
            limit=1
        )

        if results[0]:
            point = results[0][0]
            return Document(
                page_content=point.payload.get("page_content", ""),
                metadata=point.payload.get("metadata", {})
            )

        return None
