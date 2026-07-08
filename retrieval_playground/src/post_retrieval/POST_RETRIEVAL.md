# Post-Retrieval Processing

This module prepares retrieved chunks **after** retrieval and reranking, before generation.

## Contents

### Context Preparation (`3_Post_Retrieval.ipynb`)

Three independent components that plug in after retrieval:

| Component | Module | Primary Target | Latency |
|-----------|--------|----------------|---------|
| **Retrieval Grading** | `retrieval_grading.py` | False-positive chunks | Low–Medium |
| **Knowledge Refinement** | `knowledge_refinement.py` | Within-chunk filler | Medium |
| **Context Compression** | `context_compression.py` | Noise + context budget | Medium |

`context_preparation.py` chains all three into a single pipeline.

### Document Assembly

`document_assembly.py` provides the default production pattern: concatenate prepared chunks and generate one answer (stuff chain).

Older multi-pass strategies (refine, map-reduce, map-rerank) addressed context limits that rarely bind with modern models.

## Quick Start

```python
from retrieval_playground.src.post_retrieval import (
    RetrievalGrader,
    KnowledgeRefiner,
    ContextCompressor,
    ContextPreparer,
)
from retrieval_playground.src.post_retrieval import document_assembly

# Grade → refine → compress
preparer = ContextPreparer()
result = preparer.prepare(question, retrieved_chunks)

# Assemble and generate
answer = document_assembly.generate_answer(question, result.chunks)
```

## Notebook

`retrieval_playground/tutorial/3_Post_Retrieval.ipynb` — full walkthrough with evaluation on the shared test queries.
