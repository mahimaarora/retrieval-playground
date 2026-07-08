"""
Chunking strategies module.

This module provides 4 core chunking strategies:
- Recursive Character: Simple baseline
- Docling: Structure-aware + multimodal
- Parent-Child: Production pattern
- Contextual: LLM-enhanced
"""

from .recursive_chunking import RecursiveChunking
from .docling_chunking import DoclingChunking
from .parent_child_chunking import ParentChildChunking
from .contextual_chunking import ContextualChunking

__all__ = [
    'RecursiveChunking',
    'DoclingChunking',
    'ParentChildChunking',
    'ContextualChunking',
]
