#!/usr/bin/env python3
"""Repository configuration and settings."""

import sys
import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application configuration with environment variable loading and validation."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    python_log_level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info",
        description="Python logging level"
    )

    model_name: str = Field(
        default="gemini-3.1-flash-lite",
        description="LLM model name"
    )

    embedding_model_name: str = Field(
        default="models/gemini-embedding-2",
        description="Embedding model name"
    )

    reranker_model: str = Field(
        default="ms-marco-MiniLM-L-12-v2",
        description="Reranker model name"
    )

    default_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature"
    )

    default_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum API retries"
    )

    default_timeout: Optional[int] = Field(
        default=None,
        description="API timeout in seconds"
    )

    google_api_key: str = Field(
        ...,
        description="Google Gemini API key"
    )

    qdrant_key: str = Field(
        ...,
        description="Qdrant Cloud API key"
    )

    qdrant_url: str = Field(
        ...,
        description="Qdrant Cloud URL"
    )

    grpc_verbosity: str = Field(
        default="ERROR",
        description="gRPC verbosity level"
    )

    tokenizers_parallelism: str = Field(
        default="false",
        description="Tokenizers parallelism"
    )

    @property
    def supported_collections(self) -> list[str]:
        return [
            "recursive_character",
            "parent_child",
            "contextual",
            "docling",
            "hybrid"
        ]

    @field_validator("google_api_key", "qdrant_key", "qdrant_url")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("Field cannot be empty")
        return v.strip()

    @field_validator("qdrant_url")
    @classmethod
    def validate_url_format(cls, v: str) -> str:
        v = v.strip()
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(f"URL must start with http:// or https://")
        return v

    def model_post_init(self, __context) -> None:
        os.environ['TOKENIZERS_PARALLELISM'] = self.tokenizers_parallelism
        os.environ['GRPC_VERBOSITY'] = self.grpc_verbosity


def setup_repo_paths():
    """Setup repository paths automatically."""
    current_path = Path(__file__).resolve().parent
    repo_root = current_path.parent.parent

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    return repo_root


_settings: Optional[AppSettings] = None

def get_settings() -> AppSettings:
    global _settings
    if _settings is None:
        try:
            repo_root = setup_repo_paths()
            env_file = repo_root / ".env"
            _settings = AppSettings(_env_file=str(env_file) if env_file.exists() else None)
        except Exception as e:
            error_msg = (
                f"\n❌ Configuration Error: {e}\n\n"
                "Required environment variables in .env file:\n"
                "  • GOOGLE_API_KEY\n"
                "  • QDRANT_KEY\n"
                "  • QDRANT_URL\n"
            )
            raise RuntimeError(error_msg) from e
    return _settings


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if needed."""
    path.mkdir(parents=True, exist_ok=True)
    return path


REPO_ROOT = setup_repo_paths()
settings = get_settings()

PACKAGE_DIR = REPO_ROOT / "retrieval_playground"
DATA_DIR = REPO_ROOT / "data"
QDRANT_DIR = DATA_DIR / "qdrant_db"
WORKSHOP_DATA_DIR = DATA_DIR / "workshop_data"
TEST_DATA_DIR = DATA_DIR / "test_data"
TESTS_DIR = PACKAGE_DIR / "tests"
UTILS_DIR = PACKAGE_DIR / "utils"
RESULTS_DIR = DATA_DIR / "results"

MODELS_DIR = DATA_DIR / "models"
FLASHRANK_CACHE_DIR = MODELS_DIR / "flashrank"

DOCLING_IMAGES_DIR = DATA_DIR / "images"
DOCLING_IMAGES_TEMP_DIR = DATA_DIR / "images_temp"

MODEL_NAME = settings.model_name
EMBEDDING_MODEL_NAME = settings.embedding_model_name
RERANKER_MODEL = settings.reranker_model
GOOGLE_API_KEY = settings.google_api_key
QDRANT_KEY = settings.qdrant_key
QDRANT_URL = settings.qdrant_url
PYTHON_LOG_LEVEL = settings.python_log_level
GRPC_VERBOSITY = settings.grpc_verbosity
TOKENIZERS_PARALLELISM = settings.tokenizers_parallelism
DEFAULT_TEMPERATURE = settings.default_temperature
DEFAULT_MAX_RETRIES = settings.default_max_retries
DEFAULT_TIMEOUT = settings.default_timeout
SUPPORTED_COLLECTIONS = settings.supported_collections
