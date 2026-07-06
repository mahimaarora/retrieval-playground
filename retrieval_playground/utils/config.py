#!/usr/bin/env python3
"""
Repository configuration module.
Sets up base paths and common configurations for the retrieval-playground repository.
This module automatically configures Python paths when imported from anywhere in the repo.
"""

import sys
from pathlib import Path

def setup_repo_paths():
    """Setup repository paths automatically."""
    # Find the repository root by looking for specific marker files
    current_path = Path(__file__).resolve().parent
    
    # Go up two levels since config.py is now in retrieval_playground/utils/ subdirectory
    repo_root = current_path.parent.parent
    
    # Add to Python path if not already there
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    return repo_root

# Automatically setup paths when this module is imported
REPO_ROOT = setup_repo_paths()

# Common paths within the repository
PACKAGE_DIR = REPO_ROOT / "retrieval_playground"
DATA_DIR = REPO_ROOT / "data"  # Data directory at repo root level
QDRANT_DIR = DATA_DIR / "qdrant_db"
WORKSHOP_DATA_DIR = DATA_DIR / "workshop_data"  # Workshop PDFs directory
TEST_DATA_DIR = DATA_DIR / "test_data"  # Test queries and evaluation data
TESTS_DIR = PACKAGE_DIR / "tests"
UTILS_DIR = PACKAGE_DIR / "utils"
RESULTS_DIR = DATA_DIR / "results"

# Model cache directories
MODELS_DIR = DATA_DIR / "models"
FLASHRANK_CACHE_DIR = MODELS_DIR / "flashrank"  # FlashRank reranker models

# Docling-specific paths
DOCLING_IMAGES_DIR = DATA_DIR / "images"  # Permanent storage for production
DOCLING_IMAGES_TEMP_DIR = DATA_DIR / "images_temp"  # Temporary storage (auto-cleaned)

# Ensure data directories exist
DATA_DIR.mkdir(exist_ok=True)
WORKSHOP_DATA_DIR.mkdir(exist_ok=True)
TEST_DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
FLASHRANK_CACHE_DIR.mkdir(exist_ok=True)
DOCLING_IMAGES_DIR.mkdir(exist_ok=True)