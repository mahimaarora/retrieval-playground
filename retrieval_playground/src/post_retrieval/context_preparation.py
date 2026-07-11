"""
End-to-end post-retrieval context preparation pipeline.

Chains grading → refinement → compression as independent, composable steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.documents import Document

from retrieval_playground.src.post_retrieval.context_compression import ContextCompressor
from retrieval_playground.src.post_retrieval.knowledge_refinement import KnowledgeRefiner
from retrieval_playground.src.post_retrieval.retrieval_grading import RetrievalGrader


@dataclass
class PreparationResult:
    """Artifacts from a full context-preparation run."""

    chunks: List[Document]
    grading_report: List[Dict]
    token_before: int
    token_after: int


class ContextPreparer:
    """Run grading, refinement, and compression in sequence."""

    def __init__(
        self,
        grader: Optional[RetrievalGrader] = None,
        refiner: Optional[KnowledgeRefiner] = None,
        compressor: Optional[ContextCompressor] = None,
        run_refinement: bool = True,
        run_compression: bool = True,
        compression_method: str = "embedding",
    ):
        self.grader = grader or RetrievalGrader()
        self.refiner = refiner or KnowledgeRefiner()
        self.compressor = compressor or ContextCompressor()
        self.run_refinement = run_refinement
        self.run_compression = run_compression
        self.compression_method = compression_method

    def prepare(self, question: str, chunks: List[Document]) -> PreparationResult:
        """Filter, refine, and compress retrieved chunks."""
        token_before = sum(
            self.compressor.token_estimate(chunk.page_content) for chunk in chunks
        )

        graded_chunks, grading_report = self.grader.filter_chunks(question, chunks)
        prepared = graded_chunks

        if self.run_refinement:
            prepared = self.refiner.refine_chunks(question, prepared)

        if self.run_compression:
            prepared = self.compressor.compress_chunks(
                question,
                prepared,
                method=self.compression_method,
            )

        token_after = sum(
            self.compressor.token_estimate(chunk.page_content) for chunk in prepared
        )

        return PreparationResult(
            chunks=prepared,
            grading_report=grading_report,
            token_before=token_before,
            token_after=token_after,
        )
