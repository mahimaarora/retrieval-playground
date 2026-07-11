from __future__ import annotations

from typing import Optional

from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from retrieval_playground.utils.pylogger import get_python_logger
from retrieval_playground.utils import config
from semantic_router.encoders import DenseEncoder
from flashrank import Ranker
import numpy as np


def message_content(message: AIMessage | str) -> str:
    """Normalize Gemini/LangChain message content to plain text."""
    if isinstance(message, str):
        return message
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


class NormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """
    Gemini chat model that normalizes list-style response content on invoke.

    Replaces the old `base_llm | RunnableLambda(extract_text)` pattern, which
    returned a RunnableSequence without chat-model methods such as
    `with_structured_output` (post-retrieval grading) or `agenerate_prompt`
    (RAGAS). This subclass keeps those APIs while still fixing Gemini content
    for pre/mid-retrieval call sites that read `response.content`.
    """

    @staticmethod
    def _normalize_message(message: AIMessage) -> AIMessage:
        if isinstance(message.content, list):
            message.content = message_content(message)
        return message

    def invoke(self, input, config=None, **kwargs):
        return self._normalize_message(super().invoke(input, config=config, **kwargs))

    async def ainvoke(self, input, config=None, **kwargs):
        return self._normalize_message(await super().ainvoke(input, config=config, **kwargs))


class ModelManager:
    """
    Singleton model manager that provides shared instances of LLM and embeddings models.
    """

    _instance: Optional["ModelManager"] = None
    _llm: Optional[NormalizedChatGoogleGenerativeAI] = None
    _embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
    _routing_encoder = None
    _reranker: Optional[Ranker] = None  # Cached reranker instance
    _logger = get_python_logger(log_level="info")

    def __new__(cls) -> "ModelManager":
        """Ensure only one instance of ModelManager exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._logger = get_python_logger(
                log_level=config.PYTHON_LOG_LEVEL
            )
        return cls._instance

    def get_llm(self) -> NormalizedChatGoogleGenerativeAI:
        """
        Get the shared chat model.

        Works for:
        - Plain invoke / chains (pre & mid retrieval notebooks)
        - with_structured_output (post-retrieval grading)
        - RAGAS evaluation (agenerate_prompt)
        """
        if self._llm is None:
            self._initialize_models()

        return self._llm

    def get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """
        Get shared instances of embeddings model.

        Returns:
            GoogleGenerativeAIEmbeddings:
                Shared embeddings model instance
        """
        if self._embeddings is None:
            self._initialize_models()

        return self._embeddings

    def _initialize_models(self) -> None:
        """Initialize the shared model instances."""
        try:
            self._logger.info("🔄 ModelManager: Initializing shared AI models...")

            self._llm = NormalizedChatGoogleGenerativeAI(
                model=config.MODEL_NAME,
                temperature=0.1,
                max_tokens=None,
                timeout=None,
                max_retries=3,
            )

            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=config.EMBEDDING_MODEL_NAME
            )

            self._logger.info(
                "✅ ModelManager: Shared AI models initialized successfully"
            )

        except Exception as e:
            self._logger.error(f"❌ ModelManager: Model initialization failed: {e}")
            raise

    def reload_models(self) -> None:
        """Force reload of models (useful for testing or configuration changes)."""
        self._logger.info("🔄 ModelManager: Reloading models...")
        self._llm = None
        self._embeddings = None
        self._initialize_models()

    def create_routing_encoder(self, score_threshold: float = 0.7):
        """
        Create encoder for routing using Gemini embeddings.

        Args:
            score_threshold: Similarity threshold for routing (default: 0.7)

        Returns:
            Encoder instance
        """

        embeddings = self.get_embeddings()

        class GeminiEncoder(DenseEncoder):
            def __call__(self, docs):
                if isinstance(docs, str):
                    docs = [docs]
                return np.array(embeddings.embed_documents(docs))

        self._routing_encoder = GeminiEncoder(
            name=config.EMBEDDING_MODEL_NAME,
            score_threshold=score_threshold
        )
        return self._routing_encoder

    def destroy_routing_encoder(self):
        """Destroy routing encoder to free memory."""
        self._routing_encoder = None

    def get_reranker(self):
        """
        Get FlashRank reranker model (cached singleton).

        Uses persistent cache directory in project's data/models/flashrank/
        to avoid re-downloading on every system restart.

        Returns:
            FlashRank Ranker instance
        """
        if self._reranker is None:
            cache_dir = config.ensure_dir(config.FLASHRANK_CACHE_DIR)

            self._reranker = Ranker(
                model_name=config.RERANKER_MODEL,
                cache_dir=str(cache_dir)
            )
            self._logger.info(
                f"✅ Reranker initialized: FlashRank ({config.RERANKER_MODEL})"
                f" [cache: {cache_dir}]"
            )

        return self._reranker

# Global instance
model_manager = ModelManager()
