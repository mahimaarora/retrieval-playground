# Mid-Retrieval Processing

**What is Mid-Retrieval?** Advanced search techniques applied **during** the retrieval step to improve result quality.

---

## 📁 What's Inside

```
mid_retrieval/
├── hybrid_search.py              # BM25 + Dense vector search
├── reranking.py                  # Cross-encoder reranking
├── adaptive_retrieval.py         # Complexity-based auto-tuning
├── parent_child_retrieval.py     # Adaptive hierarchical retrieval
├── route_driven_retrieval.py     # Semantic routing + retrieval
└── multi_query_hybrid.py         # Multi-query with hybrid fusion
```

---

## 🚀 Quick Start

### Prerequisites
**Using Pre-Ingested Collections (Recommended for Students):**
- Instructors have provided Qdrant credentials with all collections ready to use
- Collections include: `recursive_character`, `hybrid`, `parent_child`
- Simply use the provided credentials in your `.env` file - no setup needed!

### Interactive Notebooks
- `tutorial/2A_Basic_Mid_Retrieval_Methods.ipynb` - Basic techniques (Dense, Hybrid, Reranking, Parent-Child)
- `tutorial/2B_Advanced_Mid_Retrieval_Methods.ipynb` - Advanced patterns (Multi-Query, Routing, Adaptive)

---

## 🔍 Available Techniques

### 1. **Dense Search**
Semantic similarity using vector embeddings (cosine distance).

**Usage:**
```python
from langchain_qdrant import QdrantVectorStore
from retrieval_playground.utils.model_manager import model_manager

embeddings = model_manager.get_embeddings()
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="recursive_character",
    url=config.QDRANT_URL,
    api_key=config.QDRANT_KEY
)

results = vector_store.similarity_search(query="What is BERT?", k=5)
```

**Scores:** 0-1 (higher = more similar), sorted descending

**When to use:** General semantic queries, default method

---

### 2. **Hybrid Search** (BM25 + Dense)
Combines keyword matching (BM25) with semantic search using Reciprocal Rank Fusion (RRF).

**Collection Setup (if creating your own):**
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

**Scores:** RRF formula: `1/(60 + rank + 1)`, max ~0.033, sorted descending

**When to use:** Queries with specific keywords + semantic meaning

---

### 3. **Reranking**
Two-stage retrieval: bi-encoder (fast) → cross-encoder (accurate). Uses FlashRank with ms-marco-MiniLM-L-12-v2.

```python
from retrieval_playground.src.mid_retrieval.reranking import Reranker

reranker = Reranker(
    collection_name="recursive_character",
    top_k=20,      # Initial retrieval count
    top_n=5,       # Final results after reranking
    use_cloud=True
)

# Retrieve with automatic reranking
results = reranker.retrieve(query="Explain transformers")

# Results are automatically reranked and limited to top_n (5)
for doc in results:
    print(f"Score: {doc.metadata['rerank_score']:.4f}")
    print(f"Content: {doc.page_content[:100]}...")
```

**Scores:** 0-1 (cross-encoder similarity), sorted descending

**When to use:** When top-k precision matters (RAG context, high-quality results)

---

### 4. **Parent-Child Retrieval**
Adaptive threshold-based strategy: search child chunks (1536 chars), expand to parents (6144 chars) if quality is low.

**Collection Setup (if creating your own):**
```bash
python -m retrieval_playground.utils.collection_manager parent_child --overwrite
```

**Usage:**
```python
from retrieval_playground.src.mid_retrieval.parent_child_retrieval import ParentChildRetriever

retriever = ParentChildRetriever(
    expansion_threshold=0.7,  # Expand to parents if avg_score < 0.7
    use_cloud=True
)

results = retriever.search(
    query="What is attention mechanism?",
    k=3
)
```

**How it works:**
1. Search child chunks (precise, 1536 chars ≈ 384 tokens)
2. Calculate average similarity score
3. If avg_score ≥ 0.7 → Return children (high quality)
4. If avg_score < 0.7 → Expand to parents (6144 chars ≈ 1536 tokens, 4× context)

**When to use:** Need balance between precision and context

---

### 5. **Multi-Query Hybrid**
4-stage pipeline: Generate query variants → Hybrid search each → RRF fusion → Rerank

```python
from retrieval_playground.src.mid_retrieval.multi_query_hybrid import MultiQueryHybrid

retriever = MultiQueryHybrid(collection_name="hybrid", use_cloud=True)
results = retriever.retrieve(query="Compare BERT and GPT-3", k=5)
```

**Stages:**
1. Generate 3 query variants (expand_query)
2. Hybrid search for each variant (BM25 + Dense)
3. Fuse results with RRF
4. Rerank top candidates

**When to use:** Complex queries needing multiple perspectives

---

### 6. **Route-Driven Retrieval**
Semantic routing determines retrieval strategy based on query type.

**Routes:**
- Factual → Hybrid search
- Comparison → Multi-query
- Analytical → Hybrid search
- Greetings → No retrieval

```python
from retrieval_playground.src.mid_retrieval.route_driven_retrieval import RouteDrivenRetriever

retriever = RouteDrivenRetriever(collection_name="hybrid")
results = retriever.search("Compare PyTorch vs JAX")
# Automatically routes to multi-query for comparison
```

**When to use:** Diverse query types needing different strategies

---

### 7. **Adaptive Retrieval**
Complexity-based method selection: simple → dense, moderate → hybrid, complex → multi-query

```python
from retrieval_playground.src.mid_retrieval.adaptive_retrieval import AdaptiveRetriever

retriever = AdaptiveRetriever(collection_name="hybrid")
results = retriever.search("Compare BERT and GPT")
# Analyzes complexity and selects appropriate method
```

**When to use:** Production systems with varied query types, automatic optimization

---

## 💡 Typical Workflow

```python
from retrieval_playground.utils import config
from retrieval_playground.src.mid_retrieval.hybrid_search import HybridRetriever
from retrieval_playground.src.mid_retrieval.reranking import Reranker
from retrieval_playground.src.mid_retrieval.adaptive_retrieval import AdaptiveRetriever

# Option 1: Adaptive (Recommended for Production)
# Automatically selects best method based on query complexity
retriever = AdaptiveRetriever(collection_name="hybrid")
results = retriever.search("What is RAG?")

# Option 2: Hybrid + Reranking (Manual High-Quality Pipeline)
# Two-stage: BM25+Dense → Cross-encoder reranking
reranker = Reranker(collection_name="recursive_character", top_k=20, top_n=5)
results = reranker.retrieve("What is RAG?")

# Option 3: Route-Driven (Query Type Based)
# Routes to different methods based on query intent
from retrieval_playground.src.mid_retrieval.route_driven_retrieval import RouteDrivenRetriever
route_retriever = RouteDrivenRetriever(collection_name="hybrid")
results = route_retriever.search("Compare BERT and GPT-3")

# Use in RAG pipeline
for doc in results:
    score = doc.metadata.get('rerank_score') or doc.metadata.get('score', 0)
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content[:200]}...\n")
```

---

## 📊 Performance Comparison

| Method | Recall | Precision | Latency | Complexity | Use Case |
|--------|--------|-----------|---------|------------|----------|
| **Dense** | Medium | Medium | Fast | Simple | General semantic queries |
| **Hybrid** | High | Medium | Fast | Simple | Keywords + semantics |
| **Reranking** | High | High | Moderate | Medium | High-precision RAG |
| **Parent-Child** | High | Variable | Moderate | Medium | Adaptive context |
| **Multi-Query** | High | High | High | High | Complex queries |
| **Route-Driven** | High | High | Moderate | Medium | Diverse query types |
| **Adaptive** | High | High | Moderate | Medium | Auto-optimization |

---

## 🔧 Configuration

All methods use settings from `utils/config.py`:

```python
from retrieval_playground.utils import config

# Model configuration
config.MODEL_NAME              # LLM for query generation
config.EMBEDDING_MODEL_NAME    # Bi-encoder for dense search
config.RERANKER_MODEL         # Cross-encoder for reranking

# Qdrant connection (use instructor-provided credentials)
config.QDRANT_URL
config.QDRANT_KEY
```

**Environment Setup:**
Create `.env` file with:
```bash
# Qdrant (instructor-provided)
QDRANT_URL=your_qdrant_url
QDRANT_KEY=your_qdrant_api_key

# OpenAI (for query generation)
OPENAI_API_KEY=your_openai_key
```

---

## 📦 Collection Setup

**Using Instructor-Provided Collections:**
- Use the Qdrant credentials shared by instructors
- Collections are ready: `recursive_character`, `hybrid`, `parent_child`
- No ingestion needed - just start querying!

---

## 📖 Learn More

**Interactive Tutorials:**
- `tutorial/2A_Basic_Mid_Retrieval_Methods.ipynb` - Dense, Hybrid, Reranking, Parent-Child
- `tutorial/2B_Advanced_Mid_Retrieval_Methods.ipynb` - Multi-Query, Routing, Adaptive

**Implementation Files:**
- `hybrid_search.py` - BM25 + Dense with RRF fusion
- `reranking.py` - Two-stage retrieval with cross-encoder
- `parent_child_retrieval.py` - Adaptive hierarchical retrieval
- `multi_query_hybrid.py` - 4-stage pipeline (variants → hybrid → RRF → rerank)
- `route_driven_retrieval.py` - Semantic routing based on query intent
- `adaptive_retrieval.py` - Complexity-based method selection

**Related Documentation:**
- `../pre_retrieval/` - Query preprocessing (routing, complexity analysis, rephrasing)
- `../post_retrieval/` - Post-processing techniques
- `../../SETUP.md` - Environment and collection setup guide

---

**Ready to start?** 
1. ✅ Use instructor credentials (easiest)
2. 📓 Open `tutorial/2A_Basic_Mid_Retrieval_Methods.ipynb`
3. 🚀 Run the interactive examples!

---
