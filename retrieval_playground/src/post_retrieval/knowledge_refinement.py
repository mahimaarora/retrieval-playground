"""
Knowledge refinement: strip within-chunk filler from graded passages.

Inspired by knowledge refinement in Corrective RAG (Yan et al., 2024).
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from retrieval_playground.utils.model_manager import model_manager


def split_sentences(text: str) -> List[str]:
    """Split text into sentences with a lightweight regex splitter."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


class SentenceRelevance(BaseModel):
    """Relevance decision for a single sentence."""

    relevant: bool = Field(description="True if the sentence helps answer the question.")


class KnowledgeRefiner:
    """Remove unrelated sentences and recompose tighter passages."""

    SENTENCE_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You judge whether a single sentence from a retrieved passage helps answer
the user's question. Ignore stylistic filler and tangential examples.""",
            ),
            (
                "user",
                "Question: {question}\n\nSentence: {sentence}",
            ),
        ]
    )

    PASSAGE_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Rewrite the passage so it only contains information relevant to the question.
Remove filler, tangents, and repeated ideas. Keep factual content intact.""",
            ),
            (
                "user",
                "Question: {question}\n\nPassage:\n{passage}",
            ),
        ]
    )

    def __init__(self, sentence_level: bool = True):
        self.llm = model_manager.get_llm()
        self.sentence_level = sentence_level
        self.sentence_judge = self.SENTENCE_PROMPT | self.llm.with_structured_output(
            SentenceRelevance
        )
        self.passage_refiner = self.PASSAGE_PROMPT | self.llm

    def refine_text(self, question: str, passage: str) -> str:
        """Return a tighter version of a single passage."""
        if not passage.strip():
            return passage

        if self.sentence_level:
            kept = []
            for sentence in split_sentences(passage):
                verdict = self.sentence_judge.invoke(
                    {"question": question, "sentence": sentence}
                )
                if verdict.relevant:
                    kept.append(sentence)
            if kept:
                return " ".join(kept)

        response = self.passage_refiner.invoke(
            {"question": question, "passage": passage}
        )
        return response.content.strip()

    def refine_chunks(self, question: str, chunks: List[Document]) -> List[Document]:
        """Refine each chunk in place, preserving metadata."""
        refined = []
        for chunk in chunks:
            refined.append(
                Document(
                    page_content=self.refine_text(question, chunk.page_content),
                    metadata=chunk.metadata,
                )
            )
        return refined
