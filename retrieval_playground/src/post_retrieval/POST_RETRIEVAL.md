# Post-Retrieval Processing

**What is Post-Retrieval?** Techniques applied **after** retrieval (and optional reranking) but **before** generation - to filter noise, tighten passages, and fit the context budget.

Inspired by components from Corrective RAG (Yan et al., 2024), adapted for this workshop without web-search fallback or query-rewrite loops.

---

## 📁 What's Inside

```
post_retrieval/
├── retrieval_grading.py      # LLM relevance grading (relevant / irrelevant / ambiguous)
├── knowledge_refinement.py   # Strip within-chunk filler (sentence or passage level)
├── context_compression.py    # Extractive (embedding) or abstractive (LLM) compression
├── context_preparation.py    # Chains grading → refinement → compression
├── document_assembly.py      # Stuff chain: concatenate chunks → generate answer
└── document_chain.py         # Re-exports for backward compatibility
```

---

## 🚀 Quick Start

### Prerequisites

**Environment:**
- `GOOGLE_API_KEY` - grading, refinement, and abstractive compression use Gemini
- Embeddings via `model_manager` - extractive compression uses cosine similarity
- Retrieved chunks from mid-retrieval (Tutorial 2) as `langchain_core.documents.Document` objects

**Recommended flow:**
1. Retrieve with `RAG.retrieve_context()` or mid-retrieval methods
2. Prepare context with `ContextPreparer`
3. Generate with `document_assembly.generate_answer()`
4. Measure impact in Tutorial 4 (baseline vs post-retrieval A/B)

### Interactive Notebook

- `tutorial/3_Post_Retrieval.ipynb` - operational demo: grading report, token counts, pipeline walkthrough

---

## 🔍 Available Techniques

### 1. **Retrieval Grading**

LLM judge assigns each chunk a label before it enters the generation prompt.

**Labels:** `relevant` | `irrelevant` | `ambiguous` (+ confidence + rationale)

**Usage:**
```python
from langchain_core.documents import Document
from retrieval_playground.src.post_retrieval import RetrievalGrader

grader = RetrievalGrader(confidence_threshold=0.5)
kept, report = grader.filter_chunks(
    question="What's a TPU for in ML infra?",
    chunks=retrieved_docs,
    drop_ambiguous=False,
)

for row in report:
    print(row["label"], row["confidence"], row["kept"], row["preview"])
```

**What it filters:** False-positive chunks that score high in vector search but do not help answer the question

**Latency:** Low–Medium (one LLM call per chunk)

**When to use:** Noisy retrieval, mixed-topic corpora, before expensive generation

---

### 2. **Knowledge Refinement**

Removes within-chunk filler while keeping factual content.

**Modes:**
- **Sentence-level** (default) - judge each sentence, keep relevant ones
- **Passage-level fallback** - LLM rewrite when sentence filtering is insufficient

**Usage:**
```python
from retrieval_playground.src.post_retrieval import KnowledgeRefiner

refiner = KnowledgeRefiner(sentence_level=True)
tighter_docs = refiner.refine_chunks(question, kept_chunks)
```

**What it filters:** Tangents, examples, and filler inside otherwise relevant chunks

**Latency:** Medium (one LLM call per sentence in default mode)

**When to use:** Long chunks with mixed relevance; after grading

**Trade-offs:** May drop bridging sentences; can leave dangling references - compare in Tutorial 4

---

### 3. **Context Compression**

Shrinks retained text to query-relevant spans.

**Methods:**

| Method | Mechanism | Cost |
|--------|-----------|------|
| `embedding` (default) | Cosine similarity vs query; keep top sentences | Fast, no extra LLM |
| `abstractive` | LLM summary of passage | Slower, preserves paraphrased facts |

**Usage:**
```python
from retrieval_playground.src.post_retrieval import ContextCompressor

compressor = ContextCompressor(similarity_threshold=0.35)

# Per chunk
compressed_text = compressor.compress_text(
    question, passage, method="embedding", max_sentences=3
)

# Batch
compressed_docs = compressor.compress_chunks(
    question, refined_docs, method="embedding"
)
```

**What it targets:** Context budget and residual noise after grading/refinement

**When to use:** Token limits, long passages, or when you need smaller prompts

---

### 4. **Context Preparation Pipeline**

Chains all three steps into one call.

```python
from retrieval_playground.src.post_retrieval import ContextPreparer

preparer = ContextPreparer(
    run_refinement=True,
    run_compression=True,
    compression_method="embedding",  # or "abstractive"
)
result = preparer.prepare(question, retrieved_docs)

print(f"Chunks: {len(retrieved_docs)} → {len(result.chunks)}")
print(f"Tokens: {result.token_before} → {result.token_after}")
print(result.grading_report)
```

**Returns:** `PreparationResult` with `chunks`, `grading_report`, `token_before`, `token_after`

**When to use:** Production-style default — one entry point after retrieval

---

### 5. **Document Assembly**

Concatenate prepared chunks and generate a single answer (stuff chain).

```python
from retrieval_playground.src.post_retrieval import document_assembly

answer = document_assembly.generate_answer(question, result.chunks)
```

**Pattern:** Standard production RAG — one prompt, all context stuffed in

**When to use:** After context preparation; baseline for Tutorial 4 A/B comparison

**Note:** Older multi-pass strategies (refine, map-reduce, map-rerank) addressed context limits that rarely bind with modern models — this workshop uses stuff-only.

---

## 💡 Typical Workflow

```python
from retrieval_playground.src.baseline_rag import RAG
from retrieval_playground.src.post_retrieval import ContextPreparer
from retrieval_playground.src.post_retrieval import document_assembly

rag = RAG(strategy="recursive_character")
preparer = ContextPreparer()

question = "What's a TPU for in modern ML infra?"

# 1. Retrieve (Tutorial 2)
retrieved = [doc for doc, _ in rag.retrieve_context(question, k=3)]

# 2. Prepare context (grade → refine → compress)
prepared = preparer.prepare(question, retrieved)

# 3. Generate
answer = document_assembly.generate_answer(question, prepared.chunks)

print(f"Tokens: {prepared.token_before} → {prepared.token_after}")
print(answer)
```

**Optional: step-by-step inspection**
```python
from retrieval_playground.src.post_retrieval import (
    RetrievalGrader,
    KnowledgeRefiner,
    ContextCompressor,
)

graded, report = RetrievalGrader().filter_chunks(question, retrieved)
refined = KnowledgeRefiner().refine_chunks(question, graded)
compressed = ContextCompressor().compress_chunks(question, refined)
```

---

## 📊 Technique Comparison

| Technique | Primary target | Latency | LLM calls | Best for |
|-----------|----------------|---------|-----------|----------|
| **Retrieval Grading** | Wrong chunks | Low-Medium | 1 × chunks | False positives from retrieval |
| **Knowledge Refinement** | Within-chunk noise | Medium | 1 × sentences | Long, mixed passages |
| **Compression (embedding)** | Length + noise | Fast | 0 | Token budget, quick wins |
| **Compression (abstractive)** | Length + paraphrase | Medium | 1 × chunks | Dense factual summaries |
| **ContextPreparer** | Full pipeline | Medium–High | Combined | Production default |
| **Document Assembly** | Final answer | Medium | 1 × generation | After all prep steps |

---

## 📈 Measuring Impact

Post-retrieval changes should be validated in **Tutorial 4**, not only by token counts:

| Signal | Where to check |
|--------|----------------|
| Fewer irrelevant chunks | `grading_report` in notebook 3 |
| Lower token count | `token_before` / `token_after` |
| Better context quality | RAGAS `context_precision`, `context_recall` |
| Better answers | `faithfulness`, `answer_accuracy` |
| Side effects | `answer_length_ratio`, manual answer review |

**Canonical A/B:** baseline retrieve → generate vs prepare → generate (see `4_Evaluation.ipynb`)

---

## 🔧 Configuration

All components use shared models from `utils/config.py`:

```python
from retrieval_playground.utils import config

config.MODEL_NAME           # Gemini for grading, refinement, abstractive compression
config.EMBEDDING_MODEL_NAME # Gemini embeddings for extractive compression
```

**Environment (`.env`):**
```bash
GOOGLE_API_KEY=your_gemini_api_key
```

**Tunable parameters:**

| Component | Parameter | Default | Effect |
|-----------|-----------|---------|--------|
| `RetrievalGrader` | `confidence_threshold` | 0.5 | Drop low-confidence relevant labels |
| `ContextCompressor` | `similarity_threshold` | 0.35 | Sentence keep threshold (embedding mode) |
| `ContextPreparer` | `compression_method` | `"embedding"` | `"embedding"` or `"abstractive"` |
| `ContextPreparer` | `run_refinement` | `True` | Skip refinement step if `False` |

---

## ⚠️ Production Considerations

1. **Grading drops all context** - if every chunk is irrelevant, generation has nothing to use; add a fallback policy (e.g. keep top-1 anyway).
2. **Refinement coherence** - sentence-level filtering can break flow; review answers manually on a sample.
3. **Compression assumptions** - embedding mode is extractive (keeps original sentences); abstractive mode may paraphrase.
4. **Latency stacks** - full pipeline runs multiple LLM calls; use embedding compression for workshops, abstractive selectively in production.
5. **Agent integration (Tutorial 5)** - optional: run `ContextPreparer` *inside* the retrieval tool before returning chunks to the agent.

---

**Ready to start?**
1. ✅ Confirm retrieval works (Tutorial 2) and `.env` has `GOOGLE_API_KEY`
2. 📓 Open `tutorial/3_Post_Retrieval.ipynb`
3. 🚀 Run the context preparation pipeline and inspect the grading report!

---
