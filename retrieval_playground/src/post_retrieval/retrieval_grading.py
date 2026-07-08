"""
Retrieval grading: lightweight relevance judging for retrieved chunks.

Inspired by the evaluator component in Corrective RAG (Yan et al., 2024),
without the web-search fallback or query-rewrite loop.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from retrieval_playground.utils.model_manager import model_manager

RelevanceLabel = Literal["relevant", "irrelevant", "ambiguous"]


class ChunkGrade(BaseModel):
    """Structured grade for a single retrieved chunk."""

    label: RelevanceLabel = Field(
        description="Whether the chunk helps answer the query."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the label assignment.",
    )
    rationale: str = Field(
        description="Brief explanation for the grade.",
    )


class RetrievalGrader:
    """Grade retrieved chunks before they enter the generation prompt."""

    GRADE_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a retrieval grader assessing whether a document chunk is useful
for answering a user question.

Assign one of three labels:
- relevant: the chunk contains information that helps answer the question
- irrelevant: the chunk does not help answer the question
- ambiguous: the chunk is partially related or uncertain

Also provide a confidence score between 0.0 to 1.0 based on your understanding.""",
            ),
            (
                "user",
                "Question: {question}\n\nChunk:\n{chunk}",
            ),
        ]
    )

    def __init__(self, confidence_threshold: float = 0.5):
        self.llm = model_manager.get_llm()
        self.grader = self.GRADE_PROMPT | self.llm.with_structured_output(ChunkGrade)
        self.confidence_threshold = confidence_threshold

    def grade_chunk(self, question: str, chunk: str) -> ChunkGrade:
        """Grade a single chunk for relevance to the question."""
        return self.grader.invoke({"question": question, "chunk": chunk})

    def grade_chunks(
        self, question: str, chunks: List[Document]
    ) -> List[Tuple[Document, ChunkGrade]]:
        """Grade every chunk and return (document, grade) pairs."""
        return [
            (chunk, self.grade_chunk(question, chunk.page_content))
            for chunk in chunks
        ]

    def filter_chunks(
        self,
        question: str,
        chunks: List[Document],
        drop_ambiguous: bool = False,
    ) -> Tuple[List[Document], List[Dict]]:
        """
        Drop chunks graded irrelevant (and optionally ambiguous).

        Returns:
            Tuple of filtered documents and grading metadata for inspection.
        """
        graded = self.grade_chunks(question, chunks)
        kept: List[Document] = []
        report: List[Dict] = []

        for doc, grade in graded:
            drop = grade.label == "irrelevant"
            if drop_ambiguous and grade.label == "ambiguous":
                drop = True
            if grade.label != "irrelevant" and grade.confidence < self.confidence_threshold:
                drop = True

            report.append(
                {
                    "label": grade.label,
                    "confidence": grade.confidence,
                    "rationale": grade.rationale,
                    "kept": not drop,
                    "preview": doc.page_content[:120] + "...",
                }
            )
            if not drop:
                kept.append(doc)

        return kept, report
