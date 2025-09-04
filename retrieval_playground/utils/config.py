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
DATA_DIR = PACKAGE_DIR / "data"
QDRANT_DIR = DATA_DIR / "qdrant_db"
SAMPLE_PAPERS_DIR = DATA_DIR / "sample_research_papers"
TESTS_DIR = PACKAGE_DIR / "tests"
UTILS_DIR = PACKAGE_DIR / "utils"
RESULTS_DIR = DATA_DIR / "results"

# Ensure data directories exist
DATA_DIR.mkdir(exist_ok=True)
SAMPLE_PAPERS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
