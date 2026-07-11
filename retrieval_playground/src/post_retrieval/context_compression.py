"""
Context compression: extract query-relevant spans to fit the context budget.

Uses embedding similarity for fast extractive compression; optional LLM
summarization for abstractive compression on longer passages.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieval_playground.src.post_retrieval.knowledge_refinement import split_sentences
from retrieval_playground.utils.model_manager import model_manager


class ContextCompressor:
    """Compress retained chunks by selecting query-relevant spans."""

    SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Extract only the information needed to answer the question.
Be concise but preserve key facts, names, and numbers.""",
            ),
            (
                "user",
                "Question: {question}\n\nPassage:\n{passage}",
            ),
        ]
    )

    def __init__(self, similarity_threshold: float = 0.35):
        self.embeddings = model_manager.get_embeddings()
        self.llm = model_manager.get_llm()
        self.similarity_threshold = similarity_threshold
        self.summarizer = self.SUMMARY_PROMPT | self.llm | StrOutputParser()

    def _cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        if denom == 0:
            return 0.0
        return float(np.dot(left, right) / denom)

    def _embed(self, texts: List[str]) -> List[np.ndarray]:
        vectors = self.embeddings.embed_documents(texts)
        return [np.array(vector) for vector in vectors]

    def compress_sentences(
        self,
        question: str,
        passage: str,
        max_sentences: Optional[int] = None,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Keep sentences whose embeddings are most similar to the query.

        Returns:
            Compressed passage and per-sentence similarity scores.
        """
        sentences = split_sentences(passage)
        if not sentences:
            return passage, []

        query_vector = np.array(self.embeddings.embed_query(question))
        sentence_vectors = self._embed(sentences)

        scored = [
            (sentence, self._cosine_similarity(query_vector, vector))
            for sentence, vector in zip(sentences, sentence_vectors)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)

        selected = [
            sentence
            for sentence, score in scored
            if score >= self.similarity_threshold
        ]
        if max_sentences is not None:
            selected = [sentence for sentence, _ in scored[:max_sentences]]

        if not selected:
            selected = [scored[0][0]]

        return " ".join(selected), scored

    def compress_text(
        self,
        question: str,
        passage: str,
        method: str = "embedding",
        max_sentences: Optional[int] = 3,
    ) -> str:
        """Compress a passage with embedding selection or abstractive summarization."""
        if method == "abstractive":
            return self.summarizer.invoke({"question": question, "passage": passage})

        compressed, _ = self.compress_sentences(
            question, passage, max_sentences=max_sentences
        )
        return compressed

    def compress_chunks(
        self,
        question: str,
        chunks: List[Document],
        method: str = "embedding",
        max_sentences: Optional[int] = 3,
    ) -> List[Document]:
        """Compress each chunk while preserving metadata."""
        compressed = []
        for chunk in chunks:
            compressed.append(
                Document(
                    page_content=self.compress_text(
                        question,
                        chunk.page_content,
                        method=method,
                        max_sentences=max_sentences,
                    ),
                    metadata=chunk.metadata,
                )
            )
        return compressed

    def token_estimate(self, text: str) -> int:
        """Rough token estimate for before/after comparisons in the tutorial."""
        return max(1, len(text.split()))
