# 🧩 Retrieval Playground

A Python toolkit for RAG experimentation and evaluation.

> ⚠️ **Work in Progress**: This repository is actively being developed for workshops and tutorials. Some features may be incomplete or subject to change.

## ✨ Features

### Pre-Retrieval Strategies

- **Chunking**: Baseline, recursive character, unstructured title-based, and Docling hybrid chunking  
- **Query Enhancement**: Query expansion, decomposition, rewriting, and self-querying  
- **Semantic Routing**: Route queries to appropriate knowledge domains with confidence scoring

### Mid-Retrieval Strategies

- **Basic Similarity Search**: Standard semantic search with vector databases  
- **MMR (Maximal Marginal Relevance)**: Balance relevance and diversity in results  
- **Score Thresholding**: Quality-based filtering of retrieval results  
- **Metadata Filtering**: Context-aware search with document attributes  
- **Reranking**: Cross-encoder models to reorder results for higher precision  
- **Hybrid Retrieval**: Combine BM25 keyword search with dense semantic search

### Post-Retrieval Strategies

- **Stuff Documents**: Simple concatenation of all retrieved documents  
- **Refine Chain**: Iterative refinement of answers across documents  
- **Map-Rerank**: Score and rank answers from individual documents  
- **Map-Reduce**: Summarize documents first, then combine for final answer

### Evaluation & Management

- **RAG Evaluation**: Performance benchmarking with RAGAS metrics  
- **Model Management**: Unified LLM and embedding model interfaces

## 🚀 Getting Started

### For Workshop Participants

**📁 Follow the [Setup Guides](setup-guides/) for your operating system:**

- 🍎 [Mac Setup](setup-guides/SETUP_MAC.md) - Using Docker (Recommended)
- 🪟 [Windows Setup](setup-guides/SETUP_WINDOWS.md) - Using Docker (Recommended)
- 🐧 [Linux Setup](setup-guides/SETUP_LINUX.md) - Using Docker (Recommended)
- ⚙️ [Manual Setup](setup-guides/SETUP_WITHOUT_DOCKER.md) - Without Docker (Advanced)

**Time needed:** 20-30 minutes for first-time setup

All setup guides include:

- Installation instructions
- API key configuration
- Environment verification
- Getting started with notebooks

---

## 📓 Quick Example

```python
from retrieval_playground import ModelManager
from retrieval_playground.src.pre_retrieval.chunking_strategies import PreRetrievalChunking

# Initialize and use
model_manager = ModelManager()
chunker = PreRetrievalChunking()
chunks = chunker.chunk_documents(documents, strategy="docling")
```

## 📓 Interactive Notebooks

Explore RAG techniques through hands-on Jupyter notebooks located in `retrieval_playground/tutorial/`:

- **1A_Pre_Chunking_Methods.ipynb** - Evaluate and compare different document chunking strategies using RAGAS metrics
- **1B_Pre_Query_Methods.ipynb** - Demonstrate query expansion, decomposition, rewriting, and self-querying techniques  
- **2_Mid_Retrieval_Methods.ipynb** - Explore various retrieval methods including MMR, hybrid search, and reranking
- **3_Post_Retrieval.ipynb** - Compare document chain methods for combining retrieved content into final answers

## 📄 License

MIT License