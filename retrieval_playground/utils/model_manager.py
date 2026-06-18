from __future__ import annotations

from typing import Optional
from typing import Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from retrieval_playground.utils.pylogger import get_python_logger
from retrieval_playground.utils import constants


class ModelManager:
    """
    Singleton model manager that provides shared instances of LLM and embeddings models.
    """

    _instance: Optional["ModelManager"] = None
    _llm: Optional[ChatGoogleGenerativeAI] = None
    _embeddings: Optional[GoogleGenerativeAIEmbeddings] = None  # Changed to Google embeddings
    _logger = get_python_logger(log_level="info")

    def __new__(cls) -> "ModelManager":
        """Ensure only one instance of ModelManager exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._logger = get_python_logger(
                log_level=constants.PYTHON_LOG_LEVEL
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

            # Initialize LLM
            self._llm = ChatGoogleGenerativeAI(
                model=constants.MODEL_NAME,
                temperature=0.1,
                max_tokens=None,
                timeout=None,
                max_retries=3,
            )

            # Initialize embeddings (Google Gemini)
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001"
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


# Global instance
model_manager = ModelManager()
