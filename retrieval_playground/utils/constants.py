"""
Constants for the retrieval-playground package.
"""

import os
from pathlib import Path

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file in the project root."""
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    os.environ.setdefault(key, value)

# Load .env file
load_env_file()

# Logging configuration
PYTHON_LOG_LEVEL = os.getenv("PYTHON_LOG_LEVEL", "info")

# Model configuration  
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Qdrant Configuration
QDRANT_KEY = os.getenv("QDRANT_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# Tokenizers parallelism
TOKENIZERS_PARALLELISM = os.getenv("TOKENIZERS_PARALLELISM", "false")

# API settings
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = None

# Embedding model
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL = "BAAI/bge-reranker-base"

# ALternative Lite Embedding model 
# DEFAULT_EMBEDDING_MODEL = "Alibaba-NLP/gte-multilingual-base"
# RERANKER_MODEL = "Alibaba-NLP/gte-multilingual-reranker-base"

# Data paths (relative to package)
DATA_SUBDIR = "data"
SAMPLE_PAPERS_SUBDIR = "sample_research_papers"
TESTS_SUBDIR = "tests"
UTILS_SUBDIR = "utils"
