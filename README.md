# 🧩 Retrieval Playground

A Python toolkit for RAG experimentation and evaluation.

## ✨ Features

### 🔹 Pre-Retrieval Strategies
- **Chunking**: Baseline, recursive character, unstructured title-based, and Docling hybrid chunking  
- **Query Enhancement**: Query expansion, decomposition, rewriting, and self-querying  
- **Semantic Routing**: Route queries to appropriate knowledge domains with confidence scoring  

### 🔹 Mid-Retrieval Strategies  
- **Basic Similarity Search**: Standard semantic search with vector databases  
- **MMR (Maximal Marginal Relevance)**: Balance relevance and diversity in results  
- **Score Thresholding**: Quality-based filtering of retrieval results  
- **Metadata Filtering**: Context-aware search with document attributes  
- **Reranking**: Cross-encoder models to reorder results for higher precision  
- **Hybrid Retrieval**: Combine BM25 keyword search with dense semantic search  

### 🔹 Post-Retrieval Strategies
- 📄 **Stuff Documents**: Simple concatenation of all retrieved documents  
- 🔄 **Refine Chain**: Iterative refinement of answers across documents  
- 📊 **Map-Rerank**: Score and rank answers from individual documents  
- 🔀 **Map-Reduce**: Summarize documents first, then combine for final answer  

### 🔹 Evaluation & Management
- 🧪 **RAG Evaluation**: Performance benchmarking with RAGAS metrics  
- ⚙️ **Model Management**: Unified LLM and embedding model interfaces  

## ⚡ Installation

```bash
git clone https://github.com/yourusername/retrieval-playground.git
cd retrieval-playground
pip install -e .
```

## 🚀 Quick Start

```python
from retrieval_playground import ModelManager
from retrieval_playground.src.pre_retrieval.chunking_strategies import PreRetrievalChunking

# Initialize and use
model_manager = ModelManager()
chunker = PreRetrievalChunking()
chunks = chunker.chunk_documents(documents, strategy="docling")
```

Generate test data:
```bash
rp-generate-test-data
```

## ⚙️ Configuration

```bash
export GOOGLE_API_KEY="your-google-api-key"
```

Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

## 📄 License

MIT License
