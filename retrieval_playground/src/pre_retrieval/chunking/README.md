# Chunking Strategies

Clean implementation of 4 document chunking strategies for RAG systems.

## What This Does

Breaks large documents into smaller chunks for better retrieval. Each strategy uses a different approach.

## Memory-Efficient Design

All strategies process **one PDF at a time**:
- ✅ Load one PDF → Chunk it → Push to Qdrant → Clear memory
- ✅ Prevents memory overflow with large document sets
- ✅ Immediate persistence (chunks saved right away)

## Quick Start

```python
from retrieval_playground.src.pre_retrieval.chunking import (
    ChunkingManager,
    ChunkingStrategy
)
from retrieval_playground.utils import config

# Initialize
manager = ChunkingManager()

# Run all strategies
manager.create_all_chunks(
    pdf_directory=str(config.SAMPLE_PAPERS_DIR),
    use_cloud=False
)
```

## The 4 Strategies

### 1. Recursive Character
**What:** Splits text at natural boundaries (paragraphs → sentences → words)  
**Best for:** Most documents, learning RAG  
**Speed:** Fast

```python
manager.create_chunks(
    pdf_directory="path/to/pdfs",
    strategy=ChunkingStrategy.RECURSIVE_CHARACTER
)
```

### 2. Docling
**What:** Preserves document structure, extracts images/tables  
**Best for:** Research papers, technical docs  
**Speed:** Fast

```python
manager.create_chunks(
    pdf_directory="path/to/pdfs",
    strategy=ChunkingStrategy.DOCLING
)
```

### 3. Parent-Child
**What:** Small chunks for search, large chunks for context  
**Best for:** Production RAG systems  
**Speed:** Fast

```python
manager.create_chunks(
    pdf_directory="path/to/pdfs",
    strategy=ChunkingStrategy.PARENT_CHILD
)
```

### 4. Contextual
**What:** Adds AI-generated context to each chunk  
**Best for:** Maximum accuracy, multi-document search  
**Speed:** Slow (requires LLM calls)

```python
manager.create_chunks(
    pdf_directory="path/to/pdfs",
    strategy=ChunkingStrategy.CONTEXTUAL
)
```

## File Structure

```
chunking/
├── __init__.py                  # Module exports
├── README.md                    # This file
├── base_chunking.py            # Shared utilities
├── chunking_manager.py         # Main entry point (use this!)
├── recursive_chunking.py       # Strategy 1
├── docling_chunking.py         # Strategy 2
├── parent_child_chunking.py    # Strategy 3
└── contextual_chunking.py      # Strategy 4
```

## Which Strategy to Use?

| Your Goal | Use This |
|-----------|----------|
| Learning RAG | Recursive Character |
| PDFs with images/tables | Docling |
| Production system | Parent-Child |
| Best accuracy | Contextual |

## Command Line Usage

```bash
cd retrieval_playground/src/pre_retrieval/chunking
python chunking_manager.py
```

## Modifying Chunk Size

Edit the strategy file you want to modify:

```python
# In recursive_chunking.py
self.splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,      # Change this
    chunk_overlap=100,   # Change this
)
```

## Next Steps

1. Run one strategy to test
2. Run all strategies to compare
3. Evaluate with RAGAS to find the best one for your use case

That's it! Simple and clean.
