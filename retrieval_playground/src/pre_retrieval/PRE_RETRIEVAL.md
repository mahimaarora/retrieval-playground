# Pre-Retrieval Processing

**What is Pre-Retrieval?** Everything that happens *before* you search your vector database.

Two main steps:
1. **Chunking** - Split documents into searchable pieces
2. **Query Optimization** - Transform user queries for better results

---

## 📁 What's Inside

```
pre_retrieval/
├── chunking/                    # Document chunking strategies
│   ├── recursive_chunking.py    # Simple baseline (start here)
│   ├── parent_child_chunking.py # Production-ready
│   ├── contextual_chunking.py   # LLM-enhanced
│   └── docling_chunking.py      # Handles tables & images
├── chunking_manager.py          # Run all chunking strategies
├── query_rephrasing.py          # Query transformation techniques
├── routing.py                   # Smart query routing
└── chunking_evaluation.py       # Benchmark chunking strategies
```

---

## 🚀 Quick Start

### Run Demos (See Everything in Action)

```bash
# 1. Test query rephrasing techniques
python -m retrieval_playground.src.pre_retrieval.query_rephrasing

# 2. Test semantic routing
python -m retrieval_playground.src.pre_retrieval.routing

# 3. Run chunking on workshop data
python -m retrieval_playground.src.pre_retrieval.chunking_manager
```

---

## 🧩 Chunking Strategies

**Split documents into optimal chunks for retrieval.**

### Available Strategies

| Strategy | Speed | Use Case |
|----------|-------|----------|
| **Recursive** | ⚡ Fastest | Learning, general use |
| **Parent-Child** | ⚡ Fast | Production systems |
| **Contextual** | 🐢 Moderate | High accuracy needed |
| **Docling** | 🐌 Slow | Research papers with tables/images |

### How to Run

**Run all strategies:**
```bash
python -m retrieval_playground.src.pre_retrieval.chunking_manager
```

**Run single strategy:**
```bash
python -m retrieval_playground.src.pre_retrieval.chunking_manager recursive
python -m retrieval_playground.src.pre_retrieval.chunking_manager docling
```

**Start fresh (overwrite existing):**
```bash
python -m retrieval_playground.src.pre_retrieval.chunking_manager --overwrite
```

### Use in Code

```python
from retrieval_playground.src.pre_retrieval.chunking_manager import (
    ChunkingManager, ChunkingStrategy
)

manager = ChunkingManager()

# Single strategy
manager.create_chunks(
    pdf_directory="data/workshop_data",
    strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
    use_cloud=True
)

# All strategies
manager.create_all_chunks(
    pdf_directory="data/workshop_data",
    use_cloud=True
)
```

**Recommendation:** Start with **Recursive**, upgrade to **Parent-Child** for production.

---

## 🔄 Query Rephrasing

**Transform queries to improve retrieval accuracy.**

### Available Techniques

1. **Query Expansion** - Expand abbreviations, add context
2. **Multi-Query** - Generate 3 query variants (RAG Fusion)
3. **Decomposition** - Split compound queries into parts
4. **Rewriting** - Make context-dependent queries standalone
5. **Step-Back** - Generate broader conceptual queries
6. **Complexity Classification** - Auto-detect query complexity
7. **Auto-Orchestration** - Let the system pick the best technique

### How to Run

```bash
# See all techniques in action
python -m retrieval_playground.src.pre_retrieval.query_rephrasing
```

**Example output:**
```
🔍 1. SINGLE QUERY EXPANSION
Original: What is AL?
Expanded: What does the abbreviation AL stand for?

🔍 2. MULTI-QUERY GENERATION
Original: How do quantum GNNs work?
Variants:
  1. What are the advantages of quantum GNNs?
  2. Impact of quantum graph networks
  3. Performance boost from quantum GNNs
```

### Use in Code

**Simple (recommended):**
```python
from retrieval_playground.src.pre_retrieval.query_rephrasing import (
    optimize_query_for_retrieval
)

# Automatic - picks the best technique for you
result = optimize_query_for_retrieval("Compare PyTorch and JAX")

print(result["strategy"])           # → "multi_query"
print(result["processed_queries"])  # → [variant1, variant2, variant3]
```

**Advanced (manual control):**
```python
from retrieval_playground.src.pre_retrieval.query_rephrasing import (
    expand_query, decompose_query, rewrite_query, step_back_query
)

# Single expansion
expanded = expand_query("What is ML?")
# → "What is Machine Learning?"

# Multi-query (3 variants)
variants = expand_query("How does X work?", num_variants=3)
# → [variant1, variant2, variant3]

# Decompose compound queries
parts = decompose_query("What is X and how does it work?")
# → ["What is X?", "How does X work?"]

# Rewrite with context
rewritten = rewrite_query(
    query="How effective are they?",
    previous_conversation_history="We discussed AI agents"
)
# → "How effective are AI agents in practice?"

# Step-back prompting
broader, specific = step_back_query("What CUDA optimizations improve GPUs?")
# broader → "What are fundamental GPU optimization principles?"
```

**Recommendation:** Use `optimize_query_for_retrieval()` - it handles everything automatically.

---

## 🎯 Semantic Routing

**Route queries to the best retrieval strategy.**

### Available Routes

| Route | Example Queries | Retrieval Method | Reranking |
|-------|----------------|------------------|-----------|
| **greetings** | "hi", "hello", "thanks" | None | ❌ |
| **factual** | "what is", "define" | Hybrid search | ❌ |
| **analytical** | "explain why", "how to" | Dense search | ✅ |
| **comparison** | "compare X vs Y" | Multi-query | ✅ |

### How to Run

```bash
# See routing in action
python -m retrieval_playground.src.pre_retrieval.routing
```

**Example output:**
```
Query: What is Agent Laboratory?
Detected Route: factual ✓
Confidence: 0.747
Retrieval Method: hybrid_search

Query: Compare PyTorch vs JAX
Detected Route: comparison ✓
Confidence: 0.783
Retrieval Method: multi_query
Reranking: True
```

### Use in Code

**Simple:**
```python
from retrieval_playground.src.pre_retrieval.routing import semantic_layer

result = semantic_layer("What is quantum computing?", return_metadata=True)

print(result["route_name"])        # → "factual"
print(result["retrieval_method"])  # → "hybrid_search"
print(result["use_reranking"])     # → False
```

**With Complexity Analysis:**
```python
from retrieval_playground.src.pre_retrieval.routing import (
    route_with_complexity_analysis
)

result = route_with_complexity_analysis("Compare PyTorch vs JAX")

print(result["route"]["route_name"])     # → "comparison"
print(result["complexity"]["complexity"]) # → "moderate"
print(result["final_retrieval_method"])  # → "multi_query"
```

---

## 📊 Evaluation

**Benchmark different chunking strategies.**

### How to Run

```bash
python -m retrieval_playground.src.pre_retrieval.chunking_evaluation
```

This compares all chunking strategies using RAGAS metrics and generates:
- Results CSV: `data/results/chunking_evaluation_results.csv`
- Plots: `data/results/chunking_evaluation_plots.png`

---

## 💡 Typical Workflow

### Step 1: Chunk Your Documents (One-time)

```python
from retrieval_playground.src.pre_retrieval.chunking_manager import (
    ChunkingManager, ChunkingStrategy
)

manager = ChunkingManager()
manager.create_chunks(
    pdf_directory="data/your_pdfs",
    strategy=ChunkingStrategy.PARENT_CHILD,  # Recommended
    use_cloud=True
)
```

### Step 2: Process User Queries (Every query)

```python
from retrieval_playground.src.pre_retrieval.query_rephrasing import (
    optimize_query_for_retrieval
)

user_query = "Compare PyTorch and JAX"
result = optimize_query_for_retrieval(user_query)

# Use these for retrieval
queries_to_search = result["processed_queries"]
```

### Step 3: Retrieve

Use the processed queries to search your vector database (covered in other modules).

---

## ❓ FAQ

**Q: Which chunking strategy should I start with?**  
A: **Recursive** - it's fast and simple. Great for learning.

**Q: When should I upgrade chunking strategies?**  
A: Move to **Parent-Child** for production systems. Use **Docling** if you have tables/images.

**Q: Do I need to use query rephrasing?**  
A: Yes! It significantly improves retrieval. Use `optimize_query_for_retrieval()`.

**Q: What if routing confidence is low?**  
A: Normal for ambiguous queries. The system uses sensible defaults.

**Q: Can I customize routes?**  
A: Yes! Edit `routing.py` to add/remove routes or adjust utterances.

---

## 📖 Learn More

**Tutorials:**
- `tutorial/1A_Pre_Chunking_Methods.ipynb` - Hands-on chunking guide

**Key Files:**
- `chunking_manager.py` - Chunking orchestration
- `query_rephrasing.py` - Query transformation
- `routing.py` - Query routing
- `chunking_evaluation.py` - Benchmarking

**Configuration:**
- `utils/model_manager.py` - LLM and embedding models
- `utils/config.py` - Settings and paths
- `utils/constants.py` - Default values

---

## 🎓 Quick Reference

### Commands
```bash
# Chunking
python -m retrieval_playground.src.pre_retrieval.chunking_manager
python -m retrieval_playground.src.pre_retrieval.chunking_manager recursive --overwrite

# Query rephrasing
python -m retrieval_playground.src.pre_retrieval.query_rephrasing

# Routing
python -m retrieval_playground.src.pre_retrieval.routing

# Evaluation
python -m retrieval_playground.src.pre_retrieval.chunking_evaluation
```

### Imports
```python
# Chunking
from retrieval_playground.src.pre_retrieval.chunking_manager import (
    ChunkingManager, ChunkingStrategy
)

# Query optimization (recommended)
from retrieval_playground.src.pre_retrieval.query_rephrasing import (
    optimize_query_for_retrieval
)

# Routing
from retrieval_playground.src.pre_retrieval.routing import (
    semantic_layer, route_with_complexity_analysis
)
```

---

**Ready to start?** Run `python -m retrieval_playground.src.pre_retrieval.query_rephrasing` to see query techniques in action! 🚀
