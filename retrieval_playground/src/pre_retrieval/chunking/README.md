# Chunking Strategies

Clean implementation of 4 document chunking strategies for RAG systems.

## What This Does

Breaks large documents into smaller chunks for better retrieval. Each strategy uses a different approach.

## Quick Start

```bash
# Run single strategy
python -m retrieval_playground.utils.collection_manager recursive --overwrite

# Run all strategies
python -m retrieval_playground.utils.collection_manager all --overwrite
```

## The Strategies

### 1. Recursive
**What:** Splits text at natural boundaries (paragraphs → sentences → words)  
**Best for:** Most documents, learning RAG  
**Speed:** Fast

```bash
python -m retrieval_playground.utils.collection_manager recursive --overwrite
```

### 2. Docling
**What:** Preserves document structure, extracts images/tables  
**Best for:** Research papers, technical docs  
**Speed:** Moderate

```bash
python -m retrieval_playground.utils.collection_manager docling --overwrite
```

### 3. Parent-Child
**What:** Small chunks for search, large chunks for context  
**Best for:** Production RAG systems  
**Speed:** Fast

```bash
python -m retrieval_playground.utils.collection_manager parent_child --overwrite
```

### 4. Contextual
**What:** Adds AI-generated context to each chunk  
**Best for:** Maximum accuracy, multi-document search  
**Speed:** Slow (requires LLM calls)

```bash
python -m retrieval_playground.utils.collection_manager contextual --overwrite
```

### 5. Hybrid
**What:** Adds BM25 keyword search to recursive collection  
**Best for:** Queries with specific keywords + semantic meaning  
**Speed:** Fast (copies existing collection)

```bash
# Requires recursive collection to exist first
python -m retrieval_playground.utils.collection_manager recursive --overwrite
python -m retrieval_playground.utils.collection_manager hybrid --overwrite
```

## File Structure

```
chunking/
├── __init__.py                  # Module exports
├── README.md                    # This file
├── base_chunking.py            # Shared utilities
├── recursive_chunking.py       # Strategy 1
├── docling_chunking.py         # Strategy 2
├── parent_child_chunking.py    # Strategy 3
└── contextual_chunking.py      # Strategy 4

utils/
└── collection_manager.py       # Ingestion entry point (use this!)
```

## Which Strategy to Use?

| Your Goal | Use This |
|-----------|----------|
| Learning RAG | recursive |
| PDFs with images/tables | docling |
| Production system | parent_child |
| Best accuracy | contextual |
| Keyword + semantic search | hybrid |
| Try everything | all |

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
