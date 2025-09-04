# Pre-Retrieval Processing

This module contains techniques to optimize queries and documents **before** the retrieval step in RAG systems.

## 📁 Contents

### 🧩 **Chunking Strategies** (`chunking_strategies.py`)
Different approaches to split documents into retrievable chunks:
- **Baseline**: Simple character-based splitting
- **Recursive Character**: Hierarchical splitting with overlap  
- **Unstructured**: Structure-aware chunking using titles/sections
- **Docling**: Advanced document parsing with hybrid chunking

### 🔄 **Query Rephrasing** (`query_rephrasing.py`)
Transform user queries for better retrieval:
- **Query Expansion**: Add synonyms and related terms
- **Query Decomposition**: Break complex queries into sub-queries
- **Query Rewriting**: Make context-dependent queries standalone
- **Self-Querying**: Transform natural input into optimal search queries

### 🎯 **Semantic Routing** (`routing.py`)
Route queries to appropriate knowledge domains:
- Research papers (Analytics, Computer Vision, AI, ML, Statistics)
- General knowledge and greetings
- Domain-specific routing with confidence scoring

### 📊 **Evaluation** (`chunking_evaluation.py`)
Compare and benchmark different chunking strategies using RAGAS metrics.

## 🚀 Quick Start

### Interactive Notebooks
- `Pre_1_Chunking.ipynb` - Chunking strategy evaluation demo
- `Pre_2_Rephrasing.ipynb` - Query rephrasing techniques demo  
- `Pre_3_Routing.ipynb` - Semantic routing demonstration

### Usage Example
```python
from chunking_strategies import PreRetrievalChunking, ChunkingStrategy
from query_rephrasing import expand_query, decompose_query
from routing import semantic_layer

# Process documents with different chunking strategies
chunker = PreRetrievalChunking()
chunks = chunker.chunk_documents(docs, ChunkingStrategy.DOCLING)

# Enhance queries before retrieval
expanded_query = expand_query("ML models")
sub_queries = decompose_query("What is ML and how does it work?")

# Route queries to appropriate domains
route = semantic_layer("research on computer vision")
```

## 🎯 Purpose

Pre-retrieval processing improves RAG performance by:
- **Better chunking** → More relevant document segments
- **Query enhancement** → Clearer search intent  
- **Smart routing** → Domain-appropriate retrieval
