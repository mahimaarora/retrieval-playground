# 📚 RAG Optimization Tutorial

Welcome! This tutorial teaches you how to build high-quality RAG systems through hands-on interactive notebooks covering pre-retrieval, mid-retrieval, and post-retrieval techniques.

---

## 📖 What You'll Learn

This tutorial contains **7 interactive notebooks** covering the complete RAG optimization pipeline:

**Pre-Retrieval** (Notebooks 1A & 1B)

- Optimize documents before indexing (chunking strategies)
- Transform queries before searching (expansion, routing, complexity analysis)

**Mid-Retrieval** (Notebooks 2A & 2B)

- Advanced search techniques (dense, hybrid, reranking, parent-child)
- Intelligent method selection (routing, adaptive retrieval, multi-query)

**Post-Retrieval** (Notebook 3)

- Result filtering and reranking
- Context compression and deduplication

**Evaluation & Advanced** (Notebooks 4 & 5)

- Measure and compare retrieval quality
- Build intelligent agentic RAG systems

---



## 🎯 Prerequisites



### Required

- **Google Gemini API key** — get one from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Used for query generation, contextual chunking, and expansion



### Environment Setup

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_gemini_api_key
```

---



## 🚀 How to Start



### Option 1: Using Docker (Recommended)

```bash
# Build and start the container
docker-compose up --build

# Access Jupyter at http://localhost:8888
# Token will be shown in terminal output
```

See `retrieval_playground/SETUP.md` for detailed Docker setup instructions.

### Option 2: Local Setup (Non-Docker)

```bash
# 1. Clone repository
git clone https://github.com/mahimaarora/retrieval-playground.git
cd retrieval-playground

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file (see Environment Setup above)

# 4. Launch Jupyter
jupyter notebook retrieval_playground/tutorial/
```

See `retrieval_playground/SETUP.md` for detailed local setup instructions.

---



## 📓 Notebook Sequence

Follow this order for the best learning experience:

### Pre-Retrieval (Document & Query Optimization)

**1A. Pre-Chunking Methods** 🟢

- Learn 4 chunking strategies: Recursive, Contextual, Parent-Child, Docling
- Understand trade-offs between speed, quality, and storage
- See side-by-side comparisons on real documents
- **Key Takeaway:** Start with Recursive, upgrade when needed

**1B. Pre-Query Methods** 🟢→🟡

- Query expansion, decomposition, and rewriting
- Advanced techniques: Multi-query RAG fusion, step-back prompting
- Semantic routing and complexity classification
- **Key Takeaway:** Use `optimize_query_for_retrieval()` for automatic optimization



### Mid-Retrieval (Advanced Search Techniques)

**2A. Basic Mid-Retrieval Methods**  🟡

- Dense search (semantic similarity)
- Hybrid search (BM25 + Dense with RRF fusion)
- Reranking (two-stage retrieval with cross-encoder)
- Parent-Child (adaptive hierarchical retrieval)
- **Key Takeaway:** Hybrid + Reranking for production quality

**2B. Advanced Mid-Retrieval Methods**🟡→🟠

- Multi-Query Hybrid (4-stage pipeline)
- Query Routing (intent-based method selection)
- Adaptive Retrieval (complexity-based optimization)
- **Key Takeaway:** Adaptive retrieval for automatic method selection



### Post-Retrieval (Result Optimization)

**3. Post-Retrieval Processing**  🟡

- Result filtering and deduplication
- Context compression techniques
- Final reranking and selection strategies
- **Key Takeaway:** Optimize retrieved results before sending to LLM



### Evaluation & Advanced Patterns

**4. RAG Evaluation** 🟠

- Measure retrieval quality (precision, recall, MRR)
- Evaluate end-to-end RAG performance
- Compare different retrieval strategies
- **Key Takeaway:** Measure what matters to guide optimization decisions

**5. Agentic RAG**  🟠→🔴

- Build intelligent RAG agents with LangGraph
- Implement adaptive retrieval workflows
- Create self-correcting RAG systems
- **Key Takeaway:** Combine RAG with agent patterns for complex tasks

---



## 💡 What to Expect



### Hands-On Learning

- **Visual diagrams** explain concepts before code
- **Interactive cells** let you experiment immediately
- **Before/after comparisons** show real improvements
- **Clear metrics** demonstrate quality gains



### Progressive Difficulty

- 🟢 **Basic:** Start here if you're new to RAG
- 🟡 **Intermediate:** Build on foundational knowledge
- 🟠 **Advanced:** Production-ready techniques



### Realistic Examples

- Real research papers and queries
- Actual performance metrics (+15-35% quality improvements)
- Production-ready code patterns

---



## 🔧 Pre-Ingested Collections

Collections are pre-loaded and ready to use: `recursive_character`, `hybrid`, `parent_child`, `contextual`, `docling`

- Skip ingestion steps in notebooks — start querying immediately
- All examples will work out of the box

---



## 🎓 Learning Outcomes

After completing the tutorials, you will:

✅ **Understand** all stages of RAG optimization (pre, mid, post)
✅ **Choose** the right techniques for your use case
✅ **Implement** production-quality retrieval systems
✅ **Measure** and compare retrieval quality
✅ **Optimize** for speed, quality, and cost trade-offs
✅ **Build** end-to-end RAG pipelines with confidence

---



## 🆘 Troubleshooting

**Collection not found:**

- Run `verify_setup.ipynb` to check your connection
- Collections should be pre-loaded and ready to use

**API key errors:**

- Check `.env` file exists in project root (not in `/tutorial`)
- Verify key is correct (no extra spaces or quotes)
- Ensure API key is active at [Google AI Studio](https://aistudio.google.com)

**Module not found:**

```bash
pip install -r requirements.txt
```

**Docker issues:**

- See `retrieval_playground/SETUP.md` Docker troubleshooting section
- Check logs: `docker-compose logs -f`

---

## 📚 Additional Resources

- **Setup Guide:** `retrieval_playground/SETUP.md`
- **Mid-Retrieval Docs:** `src/mid_retrieval/MID_RETRIEVAL.md`
- **Pre-Retrieval Docs:** `src/pre_retrieval/PRE_RETRIEVAL.md`
- **Source Code:** `retrieval_playground/src/`
- **GitHub Issues:** For questions and support

---



## 🎉 Ready to Start?

  
**Outcome:** Build production-ready RAG systems with confidence!

Happy learning! 🚀