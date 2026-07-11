# RAG Evaluation

**What is RAG Evaluation?** Systematic measurement of **retrieval quality**, **generation quality**, and (optionally) **agent tool selection** - applied **after** you build a pipeline so you can compare changes with evidence, not intuition.

---

## 📁 What's Inside

```
evaluation/
├── base.py                 # Shared helpers (keyword_overlap, mean_score, …)
├── ragas_runner.py         # RAGAS integration (run_ragas, RAGEvaluator)
├── retrieval_metrics.py    # Hit@k, MRR, keyword overlap + RAGAS retrieval
├── generation_metrics.py   # Faithfulness, answer relevancy, answer accuracy
├── tool_metrics.py         # Tool / routing accuracy (agentic workflows)
└── pipeline.py             # End-to-end orchestration (RAGEvaluationPipeline)
```

---

## 🚀 Quick Start

### Prerequisites

**Environment:**
- `GOOGLE_API_KEY` - Gemini LLM + embeddings (RAGAS LLM-judged metrics)
- `QDRANT_URL` / `QDRANT_KEY` - if evaluating live retrieval from indexed workshop PDFs
- RAGAS and `datasets` installed (included in workshop `requirements.txt`)

**Test dataset:**
- `data/test_data/evaluation_dataset.json` - 15 grounded Q&A pairs from workshop papers
- Each row has:
  - `user_input` - question
  - `reference_context` - gold **evidence passage** (for classical retrieval metrics)
  - `reference` - gold **answer** (for RAGAS + generation metrics)
  - `source_file` - source PDF

```python
import json
from retrieval_playground.utils import config

with open(config.TEST_DATA_DIR / "evaluation_dataset.json") as f:
    test_queries = json.load(f)
```

### Interactive Notebook

- `tutorial/4_Evaluation.ipynb` - full walkthrough: classical metrics, RAGAS, pipeline scorecard, baseline vs post-retrieval A/B

---

## 🔍 Evaluation Stages

RAG quality is not one number. Measure **each stage** separately:

```
Retrieval  →  Did we fetch the right evidence?
Generation →  Is the answer grounded and relevant?
Tools      →  Did the agent pick the right capability? (Tutorial 5)
```

---

## 📊 Available Metrics

### 1. **Classical Retrieval Metrics**

Fast, deterministic checks against **gold evidence** (`reference_context`).

| Metric | What it measures |
|--------|------------------|
| `hit_rate_at_k@3` | Did any top-k chunk overlap enough tokens with gold evidence? |
| `mrr` | How high in the ranked list does the first relevant chunk appear? |
| `keyword_overlap` | Share of reference tokens found across retrieved chunks |

**Usage:**
```python
from retrieval_playground.src.evaluation import RetrievalEvaluator

evaluator = RetrievalEvaluator(k=3)
result = evaluator.evaluate(
    questions=questions,
    contexts=retrieved_contexts,           # List[List[str]]
    reference_contexts=reference_contexts, # gold evidence
)
print(result.scores)
# {'hit_rate_at_k@3': 1.0, 'mrr': 1.0, 'keyword_overlap': 0.98}
```

**Scores:** 0-1 (higher = better)

**When to use:** Quick regression checks, debugging retrieval without LLM judge cost

---

### 2. **RAGAS Retrieval Metrics**

LLM-judged metrics comparing retrieved chunks to the gold **answer** (`reference`), not the evidence passage.

| Metric | What it measures |
|--------|------------------|
| `context_precision` | Are retrieved chunks useful for answering the question? |
| `context_recall` | Did retrieval surface the information needed for the reference answer? |

**Usage:**
```python
result = evaluator.evaluate_ragas(
    questions=questions,
    contexts=retrieved_contexts,
    reference_answers=ground_truths,  # gold ANSWERS
    answers=generated_answers,        # required by RAGAS judge
)
print(result)
# {'context_precision': 0.85, 'context_recall': 0.67}
```

**Important:** Do **not** pass `reference_context` as `reference` - RAGAS retrieval metrics expect reference **answers**.

**When to use:** Comparing retrieval strategies when answers exist; A/B baseline vs post-retrieval in Tutorial 4

---

### 3. **Generation Metrics (RAGAS + Custom)**

Evaluate the final answer against gold references and retrieved context.

| Metric | Source | What it measures |
|--------|--------|------------------|
| `faithfulness` | RAGAS | Is the answer supported by retrieved context? |
| `answer_relevancy` | RAGAS | Does the answer address the question? |
| `answer_accuracy` | RAGAS (NVIDIA) | How close is the answer to the reference answer? |
| `answer_length_ratio` | Custom | Generated length vs reference length |

**Usage:**
```python
from retrieval_playground.src.evaluation import GenerationEvaluator

gen_evaluator = GenerationEvaluator(
    ragas_metrics=["faithfulness", "answer_relevancy", "answer_accuracy"]
)
result = gen_evaluator.evaluate(
    questions=questions,
    answers=answers,
    contexts=retrieved_contexts,
    ground_truths=ground_truths,
)
print(result.ragas_scores)
print(result.custom_scores)
```

**When to use:** After changing generation prompts, post-retrieval prep, or chunk quality

---

### 4. **Tool / Agent Metrics**

Lightweight checks for agentic RAG (Tutorial 5) - did the model invoke the right tool?

| Metric | What it measures |
|--------|------------------|
| `tool_selection_accuracy` | Expected vs actual tool name |
| `retrieval_method_accuracy` | Expected vs actual retrieval method (optional) |
| `tool_call_success_rate` | Whether the tool call succeeded (optional) |

**Usage:**
```python
from retrieval_playground.src.evaluation import ToolEvaluator, ToolTrace

traces = [
    ToolTrace(
        query="What's a TPU?",
        expected_tool="retrieve_workshop_docs",
        actual_tool="retrieve_workshop_docs",
    ),
    ToolTrace(
        query="Hi!",
        expected_tool="none",
        actual_tool="none",
    ),
]
result = ToolEvaluator().evaluate(traces)
print(result.scores)
```

**When to use:** Agent notebooks; full answer quality still belongs in Tutorial 4

---

### 5. **End-to-End Pipeline**

`RAGEvaluationPipeline` runs retrieval + generation (+ optional tools) in one pass and returns a scorecard DataFrame.

```python
from retrieval_playground.src.evaluation import RAGEvaluationPipeline

pipeline = RAGEvaluationPipeline(
    retrieval_k=3,
    ragas_metrics=["faithfulness", "answer_relevancy", "answer_accuracy"],
)
result = pipeline.evaluate_rag_results(
    rag_results=rag_results,          # from RAG.query() or custom loop
    ground_truths=ground_truths,
    reference_contexts=reference_contexts,
    tool_traces=tool_traces,          # optional
)
print(result.to_dataframe())
```

**When to use:** Workshop demos, experiment logging, comparing pipeline configs

---

### 6. **Baseline vs Post-Retrieval A/B**

Canonical comparison lives in **Tutorial 4** - not in the post-retrieval notebook.

```python
from retrieval_playground.src.evaluation import RAGEvaluator, GenerationEvaluator
from retrieval_playground.src.post_retrieval import ContextPreparer, document_assembly

comparison_queries = test_queries[:2]
comparison_ground_truths = [q["reference"] for q in comparison_queries]

# Build baseline_rag_results and prepared_rag_results (see Tutorial 4)

context_evaluator = RAGEvaluator(metrics=["context_precision", "context_recall"])
gen_evaluator = GenerationEvaluator()

baseline_ctx = context_evaluator.evaluate_rag_results(baseline_rag_results, comparison_ground_truths)
prepared_ctx = context_evaluator.evaluate_rag_results(prepared_rag_results, comparison_ground_truths)
```

**Tip:** Slice `ground_truths` to match the number of queries you evaluate — batch lengths must align or RAGAS will error.

---

## 💡 Typical Workflow

```python
import json
from retrieval_playground.utils import config
from retrieval_playground.src.baseline_rag import RAG
from retrieval_playground.src.evaluation import (
    RetrievalEvaluator,
    GenerationEvaluator,
    RAGEvaluationPipeline,
)

# Load test set
with open(config.TEST_DATA_DIR / "evaluation_dataset.json") as f:
    test_queries = json.load(f)[:3]

questions = [q["user_input"] for q in test_queries]
ground_truths = [q["reference"] for q in test_queries]
reference_contexts = [q["reference_context"] for q in test_queries]

# Run RAG
rag = RAG(strategy="recursive_character")
rag_results = []
for q in test_queries:
    rag_results.append(rag.query(q["user_input"]))

answers = [r["answer"] for r in rag_results]
contexts = [[c["content"] for c in r["context"]] for r in rag_results]

# Stage 1: Classical retrieval (fast)
retrieval_eval = RetrievalEvaluator(k=3).evaluate(
    questions, contexts, reference_contexts
)
print("Classical:", retrieval_eval.scores)

# Stage 2: Full pipeline (includes RAGAS — slower, uses LLM judges)
pipeline_result = RAGEvaluationPipeline().evaluate_rag_results(
    rag_results, ground_truths, reference_contexts
)
print(pipeline_result.to_dataframe())
```

---

## 📊 Metric Comparison

| Category | Metrics | Gold label | Speed | Cost | Best for |
|----------|---------|------------|-------|------|----------|
| **Classical retrieval** | Hit@k, MRR, keyword overlap | `reference_context` | Fast | Free | Debugging retrieval |
| **RAGAS retrieval** | context_precision, context_recall | `reference` (answer) | Slow | LLM calls | Retrieval A/B |
| **RAGAS generation** | faithfulness, answer_relevancy, answer_accuracy | `reference` (answer) | Slow | LLM calls | Answer quality |
| **Custom generation** | answer_length_ratio | `reference` (answer) | Fast | Free | Verbosity checks |
| **Tools** | tool_selection_accuracy | Expected tool | Fast | Free | Agent routing |

---

## 🔧 Configuration

All evaluators use models from `utils/config.py` via `model_manager`:

```python
from retrieval_playground.utils import config

config.MODEL_NAME              # Gemini LLM (RAGAS + generation)
config.EMBEDDING_MODEL_NAME    # Embeddings (if needed by metrics)
config.TEST_DATA_DIR           # data/test_data/
config.PYTHON_LOG_LEVEL
```

**Environment (`.env`):**
```bash
GOOGLE_API_KEY=your_gemini_key
QDRANT_URL=your_qdrant_url      # for live RAG eval
QDRANT_KEY=your_qdrant_key
```

**RAGAS notes:**
- First import loads `ragas_compat` shim (Vertex AI import compatibility)
- Jupyter notebooks run RAGAS in a worker thread to avoid asyncio conflicts on Python 3.12
- Expect ~15–30 seconds per metric per query for LLM-judged scores

---

## 📦 Test Dataset

**Location:** `data/test_data/evaluation_dataset.json`

**Schema:**
```json
{
  "user_input": "why build a climate KG for AutoClimDS?",
  "reference_context": "When given the same tasks, state-of-the-art LLMs...",
  "reference": "General-purpose LLMs failed to find authoritative datasets...",
  "source_file": "2025_AutoClimDS.pdf"
}
```

**Which field for which metric:**

| Field | Used by |
|-------|---------|
| `user_input` | All stages (question) |
| `reference_context` | Classical retrieval metrics only |
| `reference` | RAGAS retrieval + generation metrics |

---

## ⚠️ Common Pitfalls

1. **Wrong gold label for RAGAS retrieval** - use `reference` (answer), not `reference_context` (evidence).
2. **Batch length mismatch** - if you evaluate 2 queries, slice `ground_truths` to 2 rows.
3. **NaN RAGAS scores** - usually API or LLM compatibility issues; restart kernel after `model_manager` updates.
4. **context_recall = 0 on small slices** - often expected when top-k chunks are short; compare trends, not absolutes.

---

Reference

- [RAGAS metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)

---

**Ready to start?**
1. ✅ Confirm `.env` has `GOOGLE_API_KEY` (and Qdrant if running live RAG)
2. 📓 Open `tutorial/4_Evaluation.ipynb`
3. 🚀 Run retrieval + generation eval on the shared test set!

---
