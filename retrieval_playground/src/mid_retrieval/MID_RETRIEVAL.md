# Mid-Retrieval Processing

**What is Mid-Retrieval?** Advanced search techniques applied **during** the retrieval step to improve result quality.

---

## 📁 What's Inside

```
mid_retrieval/
├── hybrid_search.py              # BM25 + Dense vector search
├── reranking.py                  # Cross-encoder reranking
├── adaptive_retrieval.py         # Auto-tuning retrieval
├── parent_child_retrieval.py     # Hierarchical retrieval
├── route_driven_retrieval.py     # Semantic routing + retrieval
└── multi_query_hybrid.py         # Multi-query with fusion
```

---

## 🚀 Quick Start

### Interactive Notebooks
- `tutorial/2A_Basic_Mid_Retrieval_Methods.ipynb` - Basic techniques
- `tutorial/2B_Advanced_Mid_Retrieval_Methods.ipynb` - Advanced patterns

---

## 🔍 Available Techniques

### 1. **Hybrid Search** (BM25 + Dense)
Combines keyword matching with semantic search.

**Setup (one-time):**
```bash
# Create recursive collection first
python -m retrieval_playground.utils.collection_manager recursive --overwrite

# Create hybrid collection (adds BM25 to recursive)
python -m retrieval_playground.utils.collection_manager hybrid --overwrite
```

**Usage:**
```python
from retrieval_playground.src.mid_retrieval.hybrid_search import HybridRetriever

retriever = HybridRetriever(collection_name="hybrid")
results = retriever.search(query="What is BERT?", k=5)
```

**When to use:** Queries with specific keywords + semantic meaning

---

### 2. **Reranking**
Cross-encoder models reorder results for higher precision.

```python
from retrieval_playground.src.mid_retrieval.reranking import Reranker

reranker = Reranker(
    collection_name="recursive_character",
    use_cloud=True,
    reranker_model="bge"  # or "flashrank", "huggingface"
)

results = reranker.search_and_rerank(
    query="Explain transformers",
    initial_k=20,  # Retrieve more
    final_k=5      # Return top 5 after reranking
)
```

**When to use:** When top-k precision matters (RAG context)

---

### 3. **Adaptive Retrieval**
Automatically adjusts parameters based on query complexity.

```python
from retrieval_playground.src.mid_retrieval.adaptive_retrieval import AdaptiveRetriever

retriever = AdaptiveRetriever(collection_name="hybrid")
results = retriever.search("Compare BERT and GPT")
```

**When to use:** Production systems with varied query types

---

### 4. **Parent-Child Retrieval**
Search child chunks, return parent context.

```python
from retrieval_playground.src.mid_retrieval.parent_child_retrieval import ParentChildRetriever

retriever = ParentChildRetriever(
    collection_name="parent_child",
    use_cloud=True
)

results = retriever.search(
    query="What is attention mechanism?",
    k=3
)
```

**When to use:** Need more context around relevant chunks

---

### 5. **Route-Driven Retrieval**
Semantic routing determines retrieval strategy.

```python
from retrieval_playground.src.mid_retrieval.route_driven_retrieval import RouteDrivenRetriever

retriever = RouteDrivenRetriever()
results = retriever.search("Compare PyTorch vs JAX")
```

**When to use:** Diverse query types needing different strategies

---

## 💡 Typical Workflow

```python
from retrieval_playground.utils import config
from retrieval_playground.src.mid_retrieval.hybrid_search import HybridRetriever
from retrieval_playground.src.mid_retrieval.reranking import Reranker

# 1. Hybrid Search (broad recall)
hybrid = HybridRetriever(use_cloud=True)
candidates = hybrid.search("What is RAG?", k=20)

# 2. Rerank (high precision)
reranker = Reranker(use_cloud=True)
final_results = reranker.rerank(
    query="What is RAG?",
    documents=candidates,
    top_k=5
)

# 3. Use in RAG pipeline
for doc in final_results:
    print(doc.page_content)
```

---

## 📊 Performance Comparison

| Method | Recall | Precision | Latency | Use Case |
|--------|--------|-----------|---------|----------|
| **Dense** | Medium | Medium | Fast | General queries |
| **BM25** | Low-Med | Low-Med | Fastest | Keyword queries |
| **Hybrid** | High | Medium | Fast | Most queries |
| **+ Rerank** | High | High | Moderate | Production RAG |
| **Adaptive** | High | High | Moderate | Auto-optimization |

---

## 🔧 Configuration

All methods use settings from `utils/config.py`:

```python
from retrieval_playground.utils import config

# Model configuration
config.MODEL_NAME
config.EMBEDDING_MODEL_NAME
config.RERANKER_MODEL

# Qdrant connection
config.QDRANT_URL
config.QDRANT_KEY
```

---

## 📖 Learn More

**Tutorials:**
- `tutorial/2A_Basic_Mid_Retrieval_Methods.ipynb` - Hands-on guide
- `tutorial/2B_Advanced_Mid_Retrieval_Methods.ipynb` - Advanced patterns

**Key Files:**
- `hybrid_search.py` - BM25 + Dense search
- `reranking.py` - Cross-encoder reranking
- `adaptive_retrieval.py` - Auto-tuning
- `route_driven_retrieval.py` - Semantic routing

---

**Ready to start?** Check the notebooks in `tutorial/` for interactive examples! 🚀
