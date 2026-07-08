# Mid-Retrieval Tests

Simple, beginner-friendly tests for hybrid search and reranking.

---

## Quick Start

### 1. Install Dependencies

```bash
# Required for all tests
pip install rank-bm25

# Optional: For better reranking (recommended)
pip install flashrank sentence-transformers
```

### 2. Run Tests

```bash
# Test hybrid search
python -m retrieval_playground.tests.test_hybrid_search

# Test reranking models
python -m retrieval_playground.tests.test_reranking_models

# Test full pipeline (hybrid + reranking)
python -m retrieval_playground.tests.test_hybrid_with_reranking
```

---

## Test Files

### `test_hybrid_search.py`
**What it tests:** Hybrid Search (BM25 + Dense)

**Tests:**
1. Basic hybrid search
2. BM25 vs Dense vs Hybrid comparison
3. Multiple query testing
4. Parameter tuning

**Run time:** ~2-3 minutes

**Example output:**
```
TEST 1: Basic Hybrid Search
🔍 Query: What is BERT?

📊 Results:
1. RRF Score: 0.0234 - BERT is a transformer-based model...
2. RRF Score: 0.0189 - Transformers use attention mechanisms...
✅ Test 1 passed!
```

---

### `test_reranking_models.py`
**What it tests:** Multiple reranker models

**Models tested:**
- HuggingFace (default)
- BGE Reranker v2-m3 (best free)
- FlashRank (fastest)

**Tests:**
1. Default HuggingFace reranker
2. BGE reranker
3. FlashRank reranker
4. Speed comparison
5. Different top_n values
6. Full evaluation (optional)

**Run time:** ~5-10 minutes (longer if running full evaluation)

**Example output:**
```
TEST 4: Model Comparison (Speed & Quality)
Testing huggingface...
✅ huggingface: 0.523s

Testing bge...
✅ bge: 0.187s

Testing flashrank...
✅ flashrank: 0.018s
```

---

### `test_hybrid_with_reranking.py`
**What it tests:** Full pipeline (Hybrid + Reranking)

**Pipeline:**
```
BM25 (k=50) + Dense (k=50)
    ↓
RRF Fusion (k=100)
    ↓
Reranking
    ↓
Top 5 Results
```

**Tests:**
1. Basic pipeline
2. Pipeline comparison (Dense vs Hybrid vs Hybrid+Rerank)
3. Multi-query testing
4. Different reranker models
5. Stage-by-stage impact analysis

**Run time:** ~3-5 minutes

**Example output:**
```
TEST 5: Pipeline Stage Impact Analysis
Stage 1: BM25 Only - Top Score: 15.23
Stage 2: Dense Only - Top Score: 0.89
Stage 3: Hybrid - Top RRF Score: 0.023
Stage 4: Hybrid + Reranking - Top Score: 0.94

📊 Summary: Each stage improves result quality!
```

---

## Understanding the Output

### Scores Explained

**BM25 Score** (15-20 typical)
- Keyword matching score
- Higher = better keyword match
- Not normalized (can be >1)

**Dense Score** (0-1 typical)
- Semantic similarity score
- Cosine similarity between embeddings
- 1.0 = identical, 0.0 = unrelated

**RRF Score** (0.01-0.05 typical)
- Reciprocal Rank Fusion score
- Combines BM25 + Dense rankings
- Higher = appeared in more/better positions

**Rerank Score** (0-1 or varies by model)
- Cross-encoder relevance score
- Model-specific (different ranges)
- Higher = more relevant

### What's Good?

| Method | Good Score | Great Score |
|--------|-----------|-------------|
| Dense | >0.7 | >0.85 |
| BM25 | >10 | >20 |
| RRF | >0.015 | >0.03 |
| Rerank (BGE) | >0.5 | >0.8 |
| Rerank (FlashRank) | >0.3 | >0.6 |

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'rank_bm25'`
**Fix:** `pip install rank-bm25`

### Error: `ModuleNotFoundError: No module named 'flashrank'`
**Fix:** `pip install flashrank` (optional but recommended)

### Error: `ModuleNotFoundError: No module named 'sentence_transformers'`
**Fix:** `pip install sentence-transformers` (optional, for BGE)

### Test skipped: "⚠️ BGE not available"
**Not a problem!** BGE is optional. Install with:
```bash
pip install sentence-transformers
```

### Test skipped: "⚠️ FlashRank not available"
**Not a problem!** FlashRank is optional. Install with:
```bash
pip install flashrank
```

### Qdrant connection error
**Check:**
1. Is local Qdrant running? Or using cloud?
2. Update `use_cloud=True/False` in test files
3. Check Qdrant path in config

---

## What You'll Learn

### From Hybrid Search Tests
- ✅ Why combine BM25 + Dense search
- ✅ How RRF merges multiple rankings
- ✅ When hybrid search helps most
- ✅ Parameter tuning (k values, RRF constant)

### From Reranking Tests
- ✅ Different reranker models (speed vs quality)
- ✅ When to use which model
- ✅ How much improvement to expect
- ✅ Reranking evaluation methodology

### From Combined Pipeline Tests
- ✅ How to build a complete retrieval pipeline
- ✅ Impact of each pipeline stage
- ✅ Multiplicative quality improvements
- ✅ Best practices for production systems

---

## Expected Results

### Hybrid Search
- **Improvement:** +15-25% recall vs dense-only
- **Best for:** Queries with specific keywords + semantic meaning
- **Example:** "BERT FLOPS count" (needs both "FLOPS" keyword + semantic understanding)

### Reranking
- **Improvement:** +5-18% precision on top results
- **Best for:** When top-k quality matters (RAG context)
- **Model comparison:**
  - FlashRank: Fastest (<20ms), good quality
  - BGE: Best free quality (~100ms)
  - HuggingFace: Default, reliable

### Combined Pipeline
- **Improvement:** +40-60% overall quality (multiplicative!)
- **Best for:** Production RAG systems
- **Trade-off:** +300-400ms latency (acceptable for quality gain)

---

## Next Steps

After running tests:

1. **Try on your own queries**
   - Edit test files with custom queries
   - See which pipeline works best

2. **Tune parameters**
   - Adjust k values (retrieval size)
   - Try different reranker models
   - Experiment with RRF constant

3. **Use in notebooks**
   - See `tutorial/2_Mid_Retrieval_Methods.ipynb`
   - Interactive examples with visualizations

4. **Move to integration patterns**
   - Complexity-adaptive configuration
   - Route-driven retrieval
   - Parent-child expansion

---

## Tips

### For Beginners
- Start with `test_hybrid_search.py` (simplest)
- Don't worry about optional dependencies
- Focus on understanding the output scores
- Run tests multiple times with different queries

### For Advanced Users
- Run full evaluation in `test_reranking_models.py`
- Compare all reranker models
- Tune parameters for your use case
- Measure latency vs quality trade-offs

### For Production
- Use FlashRank for speed-critical applications
- Use BGE for best free quality
- Always test on your actual data
- Benchmark before deploying

---

## Questions?

Check the main documentation or code comments. Each test file has extensive inline comments explaining what's happening!
