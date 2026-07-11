"""
Parent-Child Chunking Strategy.

WHAT: Small chunks for search, large chunks for context
HOW: Creates child chunks (retrieval) linked to parent chunks (generation)
BEST FOR: Production RAG systems, complex Q&A
"""

from pathlib import Path
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

    def chunk_pdf_directory(
        self,
        pdf_directory: str
    ) -> List[Document]:
        """
        Chunk documents using parent-child strategy.

        Args:
            pdf_directory: Path to directory containing PDF files

        Returns:
            List of Document objects (both parent and child chunks)
        """
        self.logger.info("Starting Parent-Child Chunking")

        all_chunks = []
        total_parent_chunks = 0
        total_child_chunks = 0

        def process_pdf(pdf_file: Path) -> int:
            nonlocal total_parent_chunks, total_child_chunks

            pdf_docs = self.load_pdf(pdf_file)
            self.add_metadata(pdf_docs, pdf_file)
            parent_chunks = self.parent_splitter.split_documents(pdf_docs)

            file_parent_chunks = []
            file_child_chunks = []

            for parent_idx, parent_doc in enumerate(parent_chunks):
                parent_id = f"{pdf_file.stem}_parent_{parent_idx}"

                parent_document = Document(
                    page_content=parent_doc.page_content,
                    metadata={
                        **parent_doc.metadata,
                        "chunk_type": "parent",
                        "parent_id": parent_id,
                        "parent_index": parent_idx
                    }
                )
                file_parent_chunks.append(parent_document)

                child_chunks = self.child_splitter.split_documents([parent_doc])

                for child_chunk in child_chunks:
                    child_chunk.metadata["chunk_type"] = "child"
                    child_chunk.metadata["parent_id"] = parent_id
                    child_chunk.metadata["parent_chunk_index"] = parent_idx

                self.add_chunk_ids(child_chunks)
                file_child_chunks.extend(child_chunks)

            self.add_chunk_ids(file_parent_chunks)

            all_chunks.extend(file_parent_chunks)
            all_chunks.extend(file_child_chunks)

            total_parent_chunks += len(file_parent_chunks)
            total_child_chunks += len(file_child_chunks)

            # Store count before deleting
            chunk_count = len(file_child_chunks)

            del pdf_docs
            del parent_chunks
            del file_parent_chunks
            del file_child_chunks

            return chunk_count

        self.process_pdf_directory(pdf_directory, process_pdf)

        self.logger.info(
            f"✅ Parent-child chunking complete:\n"
            f"   - {total_parent_chunks} parent chunks\n"
            f"   - {total_child_chunks} child chunks"
        )

        return all_chunks

    def get_parent_chunk(self, vector_store, parent_id: str) -> Document:
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
