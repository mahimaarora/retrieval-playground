"""Retrieval Playground - A toolkit for RAG experiments and evaluation."""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

__version__ = "0.1.0"

from .utils.model_manager import ModelManager, model_manager
from .utils.config import REPO_ROOT, DATA_DIR, SAMPLE_PAPERS_DIR
from .utils.pylogger import get_python_logger
