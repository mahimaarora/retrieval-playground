# 🧩 Retrieval Playground

A Python toolkit for RAG experimentation and evaluation.

## ✨ Features

### Pre-Retrieval Strategies

- **Chunking**: Recursive character, contextual (LLM-summary prepended), parent-child (dual-tier hierarchy), and Docling multimodal (text + tables + images)  
- **Query Enhancement**: Query expansion, multi-query / RAG Fusion, decomposition, conversational rewriting, and step-back prompting  
- **Semantic Routing**: Route queries to appropriate retrieval methods with confidence scoring  
- **Smart Automation**: Complexity classification, auto-orchestration (`optimize_query_for_retrieval`), and routing + complexity combined dispatch

### Mid-Retrieval Strategies

- **Dense Search**: Standard semantic search with vector databases  
- **Hybrid Search**: Combine BM25 keyword search with dense semantic search via Reciprocal Rank Fusion  
- **Reranking**: Two-stage retrieval with cross-encoder (FlashRank) to reorder results for higher precision  
- **Parent-Child Retrieval**: Adaptive threshold-based expansion from precise child chunks to context-rich parents  
- **Multi-Query Hybrid**: Four-stage pipeline (variant generation → hybrid search → RRF fusion → reranking)  
- **Route-Driven Retrieval**: Semantic routing to automatically select the retrieval method per query type  
- **Adaptive Retrieval**: Complexity-based auto-configuration of method, result count, and reranking

### Post-Retrieval Strategies

- **Retrieval Grading**: LLM relevance labels (relevant / irrelevant / ambiguous) to drop false-positive chunks before generation  
- **Knowledge Refinement**: Sentence- or passage-level tightening to remove filler within kept chunks  
- **Context Compression**: Extractive (embedding similarity) or abstractive (LLM summary) compression to fit token budgets  
- **Document Assembly**: Stuff-documents generation over prepared context for final answers

### Evaluation

- **Retrieval metrics**: Classical checks (hit rate@k, MRR, keyword overlap) plus RAGAS context precision/recall  
- **Generation metrics**: RAGAS faithfulness, answer relevancy, answer accuracy, plus custom length checks  
- **Tool / agent metrics**: Tool selection accuracy from logged traces (routing demo from Tutorial 1B)  
- **Custom metrics**: Extend with RAGAS `SingleTurnMetric` or lightweight Python scorers  

### Agentic RAG

- **LangGraph ReAct agent**: One retrieval tool (`retrieve_workshop_docs`) backed by the workshop vector RAG stack  
- **Prompt-based routing**: Greetings answered directly; technical questions trigger retrieval (no extra router code in the agent loop)  
- **Lightweight tool eval**: Reuse `ToolEvaluator` on agent runs without full Tutorial 4 metric suite

### Evaluation & Management

- **Model Management**: Unified LLM and embedding model interfaces (shared across all tutorials)

## 🚀 Getting Started

### For Workshop Participants

**📁 Follow the [Setup Guides](setup-guides/) for your operating system:**

- 🍎 [Mac Setup](setup-guides/SETUP_MAC.md) - Using Docker (Recommended)
- 🪟 [Windows Setup](setup-guides/SETUP_WINDOWS.md) - Using Docker (Recommended)
- 🐧 [Linux Setup](setup-guides/SETUP_LINUX.md) - Using Docker (Recommended)
- ⚙️ [Manual Setup](setup-guides/SETUP_WITHOUT_DOCKER.md) - Without Docker (Advanced)

**Time needed:** 20-30 minutes for first-time setup

All setup guides include:

- Installation instructions
- API key configuration
- Environment verification
- Getting started with notebooks

---

## 📓 Quick Example

```python
from retrieval_playground import ModelManager
from retrieval_playground.src.pre_retrieval.chunking_strategies import PreRetrievalChunking

# Initialize and use
model_manager = ModelManager()
chunker = PreRetrievalChunking()
chunks = chunker.chunk_documents(documents, strategy="docling")
```

## 📓 Interactive Notebooks

Explore RAG techniques through hands-on Jupyter notebooks in `retrieval_playground/tutorial/` (run in order):

| Notebook | Topic |
|----------|--------|
| **1A_Pre_Chunking_Methods.ipynb** | Compare chunking strategies (recursive, parent-child, contextual, Docling) |
| **1B_Pre_Query_Methods.ipynb** | Query expansion, decomposition, rewriting, multi-query, step-back, semantic routing |
| **2A_Basic_Mid_Retrieval_Methods.ipynb** | Dense search, hybrid BM25+dense search, cross-encoder reranking, parent-child retrieval |
| **2B_Advanced_Mid_Retrieval_Methods.ipynb** | Multi-query hybrid pipeline, query routing, adaptive retrieval, production pipeline comparison |
| **3_Post_Retrieval.ipynb** | Retrieval grading, knowledge refinement, context compression, full context preparation pipeline, document assembly |
| **4_Evaluation.ipynb** | Retrieval, generation, and tool metrics; RAGAS runners; custom metrics; pipeline scorecard; baseline vs post-retrieval A/B |
| **5_Agentic_RAG.ipynb** | LangGraph ReAct agent with `retrieve_workshop_docs`; prompt routing; lightweight tool-selection check |

Start with **1A**, or open `setup-guides/verify_setup.ipynb` after environment setup to confirm API keys and dependencies.

## 📄 License

MIT License