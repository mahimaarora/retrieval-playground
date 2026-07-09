"""
Contextual Chunking Strategy.

WHAT: LLM-enhanced chunks with added context
HOW: Uses AI to add contextual information before embedding
BEST FOR: Production systems, multi-document search, maximum accuracy
"""

from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from retrieval_playground.utils.model_manager import model_manager
from .base_chunking import BaseChunking


class ContextualChunking(BaseChunking):
    """
    Contextual Chunking (LLM-Enhanced).

    How it works:
    1. Create base chunks (using recursive splitting)
    2. For each chunk, ask LLM: "What's the context?"
    3. Prepend context to chunk
    4. Embed the enriched chunk

    Example:
        Original chunk:
          "Revenue: $450M (up from $390M in Q2 2024)"

        LLM adds context:
          "This chunk is from ACME Corp's Q3 2024 earnings report,
           specifically the Financial Performance section.

           ---

           Revenue: $450M (up from $390M in Q2 2024)"

    Benefits:
        - 35-67% reduction in retrieval failures
        - Chunks are self-contained (know their source)
        - Better for multi-document search

    Trade-offs:
        - Requires LLM calls (~$12 per 1000 docs with caching)
        - Slower processing
    """

    def __init__(self):
        """Initialize Contextual Chunking."""
        super().__init__(strategy_name="contextual")

        # Get LLM from model manager
        self.llm = model_manager.get_llm()

        # Create base splitter
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,             # 512 tokens (same as baseline)
            chunk_overlap=205,           # 10% overlap
            separators=["\n\n", "\n", " ", ""]
        )

        self.logger.info("✅ Contextual Chunking initialized")

    def chunk_single_pdf(self, pdf_path: str) -> List[Document]:
        """Chunk a single PDF with LLM-generated context."""
        pdf_file = Path(pdf_path)
        pdf_docs = self.load_pdf(pdf_file)

        # Generate file-level context
        full_text = "\n\n".join([doc.page_content for doc in pdf_docs])
        full_text_preview = full_text[:8000] if len(full_text) > 8000 else full_text

        try:
            file_context = self._generate_file_context(full_text_preview, pdf_file.name)
        except Exception as e:
            self.logger.warning(f"Context generation failed: {e}")
            file_context = f"This chunk is from the document: {pdf_file.name}"

        # Create base chunks
        base_chunks = self.base_splitter.split_documents(pdf_docs)

        # Add context to each chunk
        contextual_chunks = []
        for chunk in base_chunks:
            enriched_content = f"{file_context}\n\n---\n\n{chunk.page_content}"

            contextual_doc = Document(
                page_content=enriched_content,
                metadata={
                    "source": pdf_file.name,
                    "chunking_strategy": self.strategy_name
                }
            )
            contextual_chunks.append(contextual_doc)

        self.add_chunk_ids(contextual_chunks)
        return contextual_chunks

    def chunk_pdf_directory(
        self,
        pdf_directory: str
    ) -> List[Document]:
        """
        Chunk documents using contextual enhancement.

        Args:
            pdf_directory: Path to directory containing PDF files

        Returns:
            List of contextually enhanced Document objects
        """
        self.logger.info("Starting Contextual Chunking (LLM-Enhanced)")
        self.logger.info("⚠️  This will make LLM calls - expect slower processing")

        all_chunks = []

        def process_pdf(pdf_file: Path) -> int:
            pdf_docs = self.load_pdf(pdf_file)
            full_text = "\n\n".join([doc.page_content for doc in pdf_docs])
            full_text_preview = full_text[:8000] if len(full_text) > 8000 else full_text

            self.logger.info(f"    Generating file-level context using LLM...")

            try:
                file_context = self._generate_file_context(
                    full_document=full_text_preview,
                    document_name=pdf_file.name
                )
                self.logger.info(f"    ✓ Generated context for entire file")
            except Exception as e:
                self.logger.warning(f"    ⚠️  Context generation failed: {e}")
                file_context = f"This chunk is from the document: {pdf_file.name}"

            base_chunks = self.base_splitter.split_documents(pdf_docs)
            self.logger.info(f"    Adding same context to all {len(base_chunks)} chunks...")

            contextual_chunks = []
            for chunk in base_chunks:
                enriched_content = f"{file_context}\n\n---\n\n{chunk.page_content}"
                contextual_doc = Document(
                    page_content=enriched_content,
                    metadata={
                        "source": pdf_file.name,
                        "chunking_strategy": self.strategy_name
                    }
                )
                contextual_chunks.append(contextual_doc)

            self.add_chunk_ids(contextual_chunks)
            all_chunks.extend(contextual_chunks)

            chunk_count = len(contextual_chunks)
            del pdf_docs
            del full_text
            del full_text_preview
            del base_chunks
            del contextual_chunks

            return chunk_count

        total = self.process_pdf_directory(pdf_directory, process_pdf)
        self.logger.info(
            f"✅ Contextual chunking complete: {total} total chunks "
            f"(with LLM-generated context)"
        )

        return all_chunks

    def _generate_file_context(
        self,
        full_document: str,
        document_name: str
    ) -> str:
        """
        Generate contextual information for an entire file using LLM.

        This context will be added to ALL chunks from this file.

        Args:
            full_document: The complete source document (or preview)
            document_name: Name of the source file

        Returns:
            Context string (2-3 sentences) describing the document
        """

        prompt = f"""<document>
{full_document}
</document>

Please provide a short succinct context (2-3 sentences) to describe this document for the purposes of improving search retrieval.

Focus on:
- What is the main topic or subject of this document
- Key entities, concepts, or themes
- What type of document this is (research paper, report, etc.)

Answer only with the succinct context and nothing else.
"""

        try:
            # Call LLM to generate context
            response = self.llm.invoke(prompt)
            context = response.content.strip()
            return context

        except Exception as e:
            self.logger.warning(f"LLM context generation failed: {e}")
            # Fallback: simple context
            return f"This chunk is from the document: {document_name}"
