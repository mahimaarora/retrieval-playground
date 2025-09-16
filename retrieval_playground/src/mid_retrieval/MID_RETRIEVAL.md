# Mid-Retrieval Processing

This module enhances retrieval quality **during** the retrieval step through advanced search techniques and result optimization.

## 📁 Contents

### 🔄 **Reranking** (`reranking.py`)
Cross-encoder based reranking for improved result precision:
- **HuggingFace Cross-Encoder**: Advanced semantic reranking model
- **Contextual Compression**: Intelligent document filtering
- **Performance Evaluation**: Cosine similarity based benchmarking
- **Flexible Configuration**: Configurable top-k and top-n parameters

### 📊 **Retrieval Methods** (`2_Mid_Retrieval_Methods.ipynb`)
Comprehensive retrieval techniques for different use cases:
- **Basic Similarity Search**: Standard semantic search
- **MMR (Maximal Marginal Relevance)**: Balance relevance and diversity
- **Score Thresholding**: Quality-based filtering
- **Adaptive/Dynamic Retrieval**: Flexible result counts
- **Metadata Filtering**: Context-aware search
- **Document Chunk Linking**: Multi-document retrieval
- **LLM-Guided Filtering**: Intelligent pre-filtering
- **Hybrid Retrieval**: Combine keyword and semantic search

## 🚀 Quick Start

### Interactive Notebook
- `2_Mid_Retrieval_Methods.ipynb` - Comprehensive retrieval methods tutorial

### Usage Examples

#### Reranking Setup
```python
from reranking import Reranker
from pre_retrieval.chunking_strategies import ChunkingStrategy

# Initialize reranker with cloud storage
reranker = Reranker(
    strategy=ChunkingStrategy.UNSTRUCTURED,
    use_cloud=True,
    top_k=20,  # Initial retrieval count
    top_n=3    # Final reranked results
)

# Retrieve and rerank documents
results = reranker.retrieve("machine learning applications")
print(f"Retrieved {len(results)} reranked documents")

# Evaluate reranking performance
evaluation = reranker.evaluate_reranking()
print(f"Improvement: {evaluation['improvement']:.4f}")
```

#### Advanced Retrieval Methods
```python
from langchain_qdrant import QdrantVectorStore

# MMR retrieval for diversity
mmr_results = vector_store.max_marginal_relevance_search(
    query="neural networks",
    k=5,
    fetch_k=20,
    lambda_mult=0.7  # Balance relevance vs diversity
)

# Score threshold filtering
threshold_results = vector_store.similarity_search_with_score(
    query="deep learning",
    k=10,
    score_threshold=0.8  # Only high-quality matches
)

# Metadata filtering
filtered_results = vector_store.similarity_search(
    query="computer vision",
    filter={"source": "research_papers", "year": 2024}
)
```

## 🎯 Key Features

### Reranking Benefits
- **Higher Precision**: Cross-encoder models provide better relevance scoring
- **Contextual Understanding**: Deep semantic analysis of query-document pairs
- **Configurable Parameters**: Flexible top-k and top-n settings
- **Performance Tracking**: Built-in evaluation against baseline retrieval

### Retrieval Method Variety
- **Semantic Search**: Dense vector similarity matching
- **Hybrid Approaches**: Combine multiple retrieval strategies
- **Quality Control**: Score-based and metadata-based filtering
- **Diversity Optimization**: MMR for varied result sets

## 🔧 Configuration Options

### Reranker Settings
- `strategy`: Chunking strategy for vector store selection
- `top_k`: Initial retrieval count (default: 20)
- `top_n`: Final reranked results (default: 3)

### Supported Models
- **Cross-Encoder**: `ms-marco-MiniLM-L-6-v2` (configurable via constants)
- **Embeddings**: Compatible with any LangChain embedding model
- **Vector Store**: Qdrant with cosine similarity

## 📈 Performance

Mid-retrieval processing improves RAG by:
- **Better relevance** → Cross-encoder reranking
- **Result diversity** → MMR and filtering techniques
- **Quality control** → Score thresholding and metadata filtering
- **Flexible retrieval** → Multiple search strategies
