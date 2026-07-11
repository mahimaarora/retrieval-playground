"""
Compatibility shim for RAGAS with modern langchain-community.

Must live outside retrieval_playground.src.evaluation so importing it does not
trigger evaluation/__init__.py (which imports ragas before the patch runs).
"""

from __future__ import annotations

import sys
import types

_VERTEXAI_CHAT_MODULE = "langchain_community.chat_models.vertexai"
_VERTEXAI_LLM_MODULE = "langchain_community.llms.vertexai"


def _ensure_module(name: str) -> types.ModuleType:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]


def patch_ragas_vertexai_imports() -> None:
    """Register shims for removed langchain_community Vertex AI paths."""
    if _VERTEXAI_CHAT_MODULE in sys.modules:
        return

    try:
        from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
        return
    except ImportError:
        pass

    chat_shim = _ensure_module(_VERTEXAI_CHAT_MODULE)
    llm_shim = _ensure_module(_VERTEXAI_LLM_MODULE)

    try:
        from langchain_google_vertexai import ChatVertexAI, VertexAI

        chat_shim.ChatVertexAI = ChatVertexAI
        llm_shim.VertexAI = VertexAI
    except ImportError:
        chat_shim.ChatVertexAI = None
        llm_shim.VertexAI = None

    llms_pkg = _ensure_module("langchain_community.llms")
    if not hasattr(llms_pkg, "VertexAI"):
        llms_pkg.VertexAI = llm_shim.VertexAI


patch_ragas_vertexai_imports()
