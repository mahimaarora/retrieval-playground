# Pre-Retrieval Processing

This module optimizes queries and documents **before** the retrieval step in RAG systems.

## 📁 Contents

### 🧩 **Chunking Strategies** (`chunking_strategies.py`)
Advanced document chunking with multiple strategies:
- **Baseline**: Simple character-based splitting
- **Recursive Character**: Hierarchical splitting with overlap  
- **Unstructured**: Structure-aware chunking using titles/sections
- **Docling**: Advanced document parsing with hybrid chunking
- **Semantic**: Embedding-based semantic chunking 

### 🔄 **Query Rephrasing** (`query_rephrasing.py`)
Four LLM-powered query transformation techniques:
- **Query Expansion**: Add context and related terms when beneficial
- **Query Decomposition**: Break compound queries into atomic sub-queries
- **Query Rewriting**: Make context-dependent queries standalone
- **Self-Querying**: Transform complex input into optimal search queries

### 🎯 **Semantic Routing** (`routing.py`)
Intelligent query routing with confidence scoring:
- **Research Papers**: Routes academic queries (Analytics, CV, AI, ML, Statistics)
- **Greetings**: Handles casual conversation
- **Fallback**: Default handling for unrecognized queries

### 📊 **Evaluation** (`chunking_evaluation.py`)
RAGAS-based benchmarking for chunking strategies.

## 🚀 Quick Start

### Interactive Notebooks
- `retrieval_playground/tutorial/1A_Pre_Chunking_Methods.ipynb` - Chunking strategy evaluation
- `retrieval_playground/tutorial/1B_Pre_Query_Methods.ipynb` - Query rephrasing techniques

### Usage Examples
```python
from chunking_strategies import PreRetrievalChunking, ChunkingStrategy
from query_rephrasing import expand_query, decompose_query, rewrite_query
from routing import semantic_layer

# Document chunking with cloud/local storage
chunker = PreRetrievalChunking()
chunker.create_and_store_chunks("./pdfs", ChunkingStrategy.DOCLING, use_cloud=True)

# Query enhancement
expanded = expand_query("AI models")  # → "artificial intelligence models and machine learning algorithms"
sub_queries = decompose_query("What is ML and how does it work?")  # → ["What is machine learning?", "How does machine learning work?"]
standalone = rewrite_query("How does it work?", "Previous context about neural networks")

# Smart routing
routed = semantic_layer("research on computer vision")  # → Routes to research_papers with confidence score
```

## 🎯 Benefits

Pre-retrieval processing improves RAG by:
- **Smart chunking** → Better document segmentation
- **Query enhancement** → Clearer search intent  
- **Intelligent routing** → Domain-appropriate retrieval
