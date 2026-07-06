"""
Chunking Manager - Main orchestration for all chunking strategies.

This is the main entry point for using chunking strategies.
"""

from enum import Enum
from pathlib import Path
import time

from retrieval_playground.utils.model_manager import model_manager
from retrieval_playground.utils.collection_manager import CollectionManager
from retrieval_playground.utils.pylogger import get_python_logger
from retrieval_playground.utils import constants, config

from retrieval_playground.src.pre_retrieval.chunking.recursive_chunking import RecursiveChunking
from retrieval_playground.src.pre_retrieval.chunking.docling_chunking import DoclingChunking
from retrieval_playground.src.pre_retrieval.chunking.parent_child_chunking import ParentChildChunking
from retrieval_playground.src.pre_retrieval.chunking.contextual_chunking import ContextualChunking


class ChunkingStrategy(Enum):
    """Available chunking strategies.

    Order for create_all_chunks():
    1. RECURSIVE_CHARACTER - Fastest, baseline
    2. PARENT_CHILD - Fast, hierarchical
    3. CONTEXTUAL - Moderate, LLM-enhanced
    4. DOCLING - Slowest, most comprehensive (multimodal)
    """
    RECURSIVE_CHARACTER = "recursive_character"
    PARENT_CHILD = "parent_child"
    CONTEXTUAL = "contextual"
    DOCLING = "docling"


class ChunkingManager:
    """
    Manager for all chunking strategies.

    This class provides a simple interface to:
    - Create chunks using any strategy
    - Run all strategies at once
    - Manage Qdrant collections

    Usage:
        manager = ChunkingManager()

        # Single strategy
        manager.create_chunks(
            pdf_directory="path/to/pdfs",
            strategy=ChunkingStrategy.RECURSIVE_CHARACTER
        )

        # All strategies
        manager.create_all_chunks(pdf_directory="path/to/pdfs")
    """

    def __init__(self):
        """Initialize the Chunking Manager."""
        self.logger = get_python_logger(log_level=constants.PYTHON_LOG_LEVEL)
        self.embeddings = model_manager.get_embeddings()

        # Initialize collection manager
        self.collection_manager = CollectionManager(self.embeddings)

        # Initialize all strategies
        self.strategies = {
            ChunkingStrategy.RECURSIVE_CHARACTER: RecursiveChunking(),
            ChunkingStrategy.DOCLING: DoclingChunking(),
            ChunkingStrategy.PARENT_CHILD: ParentChildChunking(),
            ChunkingStrategy.CONTEXTUAL: ContextualChunking(),
        }

        self.logger.info("✅ ChunkingManager initialized with 4 strategies")

    def create_chunks(
        self,
        pdf_directory: str,
        strategy: ChunkingStrategy,
        use_cloud: bool = False,
        overwrite: bool = False
    ) -> None:
        """
        Create and store chunks using specified strategy.

        Args:
            pdf_directory: Path to directory containing PDF files
            strategy: Which chunking strategy to use
            use_cloud: Use cloud Qdrant (True) or local (False)
            overwrite: If True, delete existing collection and create new one.
                      If False, add to existing collection.
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"🚀 Creating chunks with: {strategy.value}")
        self.logger.info(f"{'='*70}\n")

        # Clear Docling images if needed
        if overwrite and strategy == ChunkingStrategy.DOCLING:
            self._clear_docling_images()

        # Create collection
        qdrant_client = self.collection_manager.create_collection(
            collection_name=strategy.value,
            use_cloud=use_cloud,
            overwrite=overwrite,
            enable_hybrid=False
        )

        # Create vector store
        vector_store = self.collection_manager.get_vector_store(
            collection_name=strategy.value,
            use_cloud=use_cloud
        )

        # Get strategy instance
        chunking_strategy = self.strategies[strategy]

        # Time the operation
        start_time = time.time()

        # Run chunking
        chunking_strategy.chunk_documents(pdf_directory, vector_store)

        # Cleanup
        elapsed_time = time.time() - start_time
        self.logger.info(f"\n⏱️  Total time: {elapsed_time:.2f} seconds")

        qdrant_client.close()
        del qdrant_client
        del vector_store

        self.logger.info(f"✅ Completed: {strategy.value}\n")

    def create_all_chunks(
        self,
        pdf_directory: str,
        use_cloud: bool = False,
        overwrite: bool = False
    ) -> None:
        """
        Create chunks using all 4 strategies.

        Args:
            pdf_directory: Path to directory containing PDF files
            use_cloud: Use cloud Qdrant (True) or local (False)
            overwrite: If True, delete existing collections and create new ones.
                      If False, add to existing collections.
        """
        storage_type = "cloud" if use_cloud else "local"

        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"🚀 Creating chunks for ALL 4 strategies")
        self.logger.info(f"📍 Storage: {storage_type} Qdrant")
        self.logger.info(f"{'='*70}\n")

        strategies = list(ChunkingStrategy)

        for idx, strategy in enumerate(strategies, 1):
            self.logger.info(f"\n[{idx}/{len(strategies)}] {strategy.value.upper()}")
            self.create_chunks(pdf_directory, strategy, use_cloud, overwrite)

        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"🎉 All {len(strategies)} strategies completed!")
        self.logger.info(f"{'='*70}\n")

    def _clear_docling_images(self):
        """Clear all images from the Docling images directory."""
        import shutil

        if config.DOCLING_IMAGES_DIR.exists():
            self.logger.info(f"🗑️  Clearing images from {config.DOCLING_IMAGES_DIR}")
            shutil.rmtree(config.DOCLING_IMAGES_DIR)
            config.DOCLING_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    def get_parent_store(self, strategy: ChunkingStrategy = ChunkingStrategy.PARENT_CHILD):
        """
        Get parent document store for parent-child strategy.

        Args:
            strategy: Should be PARENT_CHILD

        Returns:
            Dictionary of parent documents
        """
        if strategy == ChunkingStrategy.PARENT_CHILD:
            return self.strategies[strategy].parent_store
        return None


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of the Chunking Manager.

    Usage:
        python chunking_manager.py                          # Run all strategies (add to existing)
        python chunking_manager.py recursive                # Run recursive only (add to existing)
        python chunking_manager.py recursive --overwrite    # Run recursive (create new collection)
        python chunking_manager.py --overwrite              # Run all (create new collections)
    """
    import sys

    # Initialize manager
    manager = ChunkingManager()

    # Parse arguments
    args = sys.argv[1:]
    overwrite = "--overwrite" in args
    strategy_name = None

    # Remove --overwrite flag to get strategy name
    args = [arg for arg in args if arg != "--overwrite"]

    if args:
        strategy_name = args[0].lower()

    strategy_map = {
        "recursive": ChunkingStrategy.RECURSIVE_CHARACTER,
        "docling": ChunkingStrategy.DOCLING,
        "parent_child": ChunkingStrategy.PARENT_CHILD,
        "contextual": ChunkingStrategy.CONTEXTUAL,
    }

    if strategy_name:
        if strategy_name in strategy_map:
            # Run single strategy
            manager.create_chunks(
                pdf_directory=str(config.WORKSHOP_DATA_DIR),
                strategy=strategy_map[strategy_name],
                use_cloud=True,
                overwrite=overwrite
            )
        else:
            print(f"❌ Unknown strategy: {strategy_name}")
            print("Available: recursive, docling, parent_child, contextual")
            print("\nOptions:")
            print("  --overwrite    Create new collection (delete existing)")
    else:
        # No strategy specified - run all strategies
        manager.create_all_chunks(
            pdf_directory=str(config.WORKSHOP_DATA_DIR),
            use_cloud=True,
            overwrite=overwrite
        )
