from __future__ import annotations

from typing import Optional
from typing import Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from retrieval_playground.utils.pylogger import get_python_logger
from retrieval_playground.utils import config
from semantic_router.encoders import DenseEncoder
from flashrank import Ranker
import numpy as np


class ModelManager:
    """
    Singleton model manager that provides shared instances of LLM and embeddings models.
    """

    _instance: Optional["ModelManager"] = None
    _llm: Optional[ChatGoogleGenerativeAI] = None
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

    def get_llm(self) -> ChatGoogleGenerativeAI:
        """
        Get shared instances of LLM.

        Returns:
            ChatGoogleGenerativeAI:
                Shared model instances
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

            # Initialize LLM with response format handling
            base_llm = ChatGoogleGenerativeAI(
                model=config.MODEL_NAME,
                temperature=0.1,
                max_tokens=None,
                timeout=None,
                max_retries=3,
            )

            # Wrap to handle list-based content from newer Gemini models
            from langchain_core.runnables import RunnableLambda

            def extract_text(ai_message):
                """Extract text from AIMessage, handling both string and list formats."""
                content = ai_message.content
                if isinstance(content, list):
                    # New format: list of content blocks
                    text_parts = [block.get('text', '') for block in content if block.get('type') == 'text']
                    ai_message.content = ''.join(text_parts)
                return ai_message

            self._llm = base_llm | RunnableLambda(extract_text)

            # Initialize embeddings
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

        # Minimal encoder: inherit from DenseEncoder and override __call__
        class GeminiEncoder(DenseEncoder):
            def __call__(self, docs):
                if isinstance(docs, str):
                    docs = [docs]
                return np.array(embeddings.embed_documents(docs))

        # Create encoder
        self._routing_encoder = GeminiEncoder(
            name=config.EMBEDDING_MODEL_NAME,
            score_threshold=score_threshold
        )
        return self._routing_encoder

    def destroy_routing_encoder(self):
        """
        Destroy routing encoder to free memory.
        """
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
            # Ensure cache directory exists
            cache_dir = config.ensure_dir(config.FLASHRANK_CACHE_DIR)

            # Initialize with persistent cache
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
