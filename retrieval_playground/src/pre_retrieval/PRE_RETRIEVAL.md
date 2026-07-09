# Pre-Retrieval Processing

**What is Pre-Retrieval?** Everything that happens *before* you search your vector database.

Two main components:

1. **Chunking** - Split documents into searchable pieces
2. **Query Optimization** - Transform user queries for better results

---

## 📁 What's Inside

```
pre_retrieval/
├── chunking/                    # Document chunking strategies
│   ├── recursive_chunking.py    # Simple baseline (start here)
│   ├── contextual_chunking.py   # LLM-enhanced with context
│   ├── parent_child_chunking.py # Production-ready precision + context
│   └── docling_chunking.py      # Multimodal: text + tables + images
├── query_rephrasing.py          # Query transformation techniques
└── routing.py                   # Smart query routing
```

---



## 🧩 Chunking Strategies

**Split documents into optimal chunks for retrieval.**

### Available Strategies


| Strategy         | Speed | Quality | Use Case                           |
| ---------------- | ----- | ------- | ---------------------------------- |
| **Recursive**    | ⚡⚡⚡   | ⭐⭐      | General use, learning              |
| **Contextual**   | ⚡⚡    | ⭐⭐⭐     | Technical docs, multi-document     |
| **Parent-Child** | ⚡⚡    | ⭐⭐⭐⭐    | Production systems                 |
| **Docling**      | ⚡     | ⭐⭐⭐⭐    | Research papers with tables/images |


**Recommendation:** Start with **Recursive**, upgrade to **Parent-Child** for production.

### Learn More

See the interactive notebooks for hands-on examples:

- `tutorial/1A_Pre_Chunking_Methods.ipynb` - Learn all 4 strategies with code examples

To ingest your own documents and create Qdrant collections, refer to the **Setup Guide** (ingestion takes ~15-30mins for all strategies).

---



## 🔄 Query Optimization

**Transform user queries to improve retrieval accuracy.**

### Available Techniques


| Technique             | Improvement | Best For                         |
| --------------------- | ----------- | -------------------------------- |
| **Expansion**         | +10-15%     | Abbreviations, vague queries     |
| **Multi-Query**       | +15-19%     | Important queries (RAG Fusion)   |
| **Decomposition**     | +15-20%     | Multi-part questions             |
| **Rewriting**         | +20-25%     | Context-dependent queries        |
| **Step-Back**         | +10-15%     | Technical/complex queries        |
| **Auto-Optimization** | +25-35%     | Production (combines techniques) |




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

# Multi-query (3 variants for RAG Fusion)
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

**Recommendation:** Use `optimize_query_for_retrieval()` - it handles complexity analysis and technique selection automatically.

### Learn More

See the interactive notebook for hands-on examples:

- `tutorial/1B_Pre_Query_Methods.ipynb` - Learn all query optimization techniques with before/after comparisons

---



## 🎯 Semantic Routing

**Route queries to the best retrieval strategy based on intent.**

### Available Routes


| Route          | Example Queries         | Retrieval Method    | Reranking |
| -------------- | ----------------------- | ------------------- | --------- |
| **greetings**  | "hi", "hello", "thanks" | None (no retrieval) | ❌         |
| **factual**    | "what is", "define"     | Hybrid search       | ❌         |
| **analytical** | "explain why", "how to" | Dense search        | ✅         |
| **comparison** | "compare X vs Y"        | Multi-query         | ✅         |




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



### Learn More

See the routing section in:

- `tutorial/1B_Pre_Query_Methods.ipynb` - Interactive routing examples

---



## 💡 Typical Workflow



### Step 1: Learn the Techniques (One-time)

Work through the interactive notebooks:

1. `tutorial/1A_Pre_Chunking_Methods.ipynb`  - Understand chunking strategies
2. `tutorial/1B_Pre_Query_Methods.ipynb` - Learn query optimization



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

Use the processed queries to search your vector database (covered in mid-retrieval and post-retrieval modules).

---



## ❓ FAQ

**Q: Which chunking strategy should I start with?**  
A: **Recursive** - it's fast, simple, and effective for learning.

**Q: When should I upgrade chunking strategies?**  
A: Move to **Parent-Child** for production systems. Use **Docling** if your documents have tables/images.

**Q: Do I need to use query rephrasing?**  
A: Yes! It significantly improves retrieval. Use `optimize_query_for_retrieval()` for automatic optimization.

**Q: Can I customize routes?**  
A: Yes! Edit `routing.py` to add/remove routes or adjust utterances.

---



## 🎓 Quick Reference



### Key Imports

```python
# Query optimization (recommended)
from retrieval_playground.src.pre_retrieval.query_rephrasing import (
    optimize_query_for_retrieval
)

# Routing
from retrieval_playground.src.pre_retrieval.routing import (
    semantic_layer, route_with_complexity_analysis
)

# Manual techniques (advanced)
from retrieval_playground.src.pre_retrieval.query_rephrasing import (
    expand_query, decompose_query, rewrite_query, step_back_query
)
```



### Configuration Files

- `utils/model_manager.py` - LLM and embedding models
- `utils/config.py` - All settings, paths, and configuration

---

**Ready to start?** Check the notebooks in `tutorial/` for interactive, hands-on learning! 🚀