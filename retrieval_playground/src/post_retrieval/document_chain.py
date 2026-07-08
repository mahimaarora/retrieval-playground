"""Backward-compatible entry point for document assembly."""

from retrieval_playground.src.post_retrieval.document_assembly import (
    STUFF_PROMPT,
    assemble_context,
    generate_answer,
    setup_stuff_chain,
)

__all__ = [
    "STUFF_PROMPT",
    "assemble_context",
    "generate_answer",
    "setup_stuff_chain",
]
