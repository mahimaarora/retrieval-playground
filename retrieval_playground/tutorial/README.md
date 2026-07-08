# 📚 Interactive Tutorial Notebooks

Welcome to the RAG optimization tutorial! These notebooks teach you pre-retrieval techniques through hands-on experimentation.

## 🎯 Learning Path

### Start Here: Prerequisites
- ✅ Basic Python knowledge
- ✅ Familiarity with RAG concepts (optional but helpful)
- ✅ ~45 minutes total time

### Notebooks (Complete in Order)

#### 1️⃣ Document Chunking (`1A_Pre_Chunking_Methods.ipynb`)
**Time:** ~20 minutes | **Difficulty:** 🟢 Beginner

Learn how to split documents for better retrieval:
- 🟢 **Recursive** - Simple & effective (your default)
- 🟡 **Contextual** - Adds LLM-powered context
- 🟠 **Parent-Child** - Precision + context
- 🔴 **Docling** - Handles tables & images

**What You'll Build:**
- Chunk a research paper 4 different ways
- Compare results side-by-side
- Learn when to use each method

**Key Takeaway:** Start with Recursive, upgrade only when needed!

---

#### 2️⃣ Query Optimization (`1B_Pre_Query_Methods.ipynb`)
**Time:** ~25 minutes | **Difficulty:** 🟢 Beginner → 🟡 Intermediate

Transform messy queries into perfect searches:

**Part 1 - Basic Transforms** (🟢)
- Expansion - Fix abbreviations
- Decomposition - Split multi-questions
- Rewriting - Add context

**Part 2 - Advanced Techniques** (🟡)
- Multi-Query RAG Fusion (+15-19% quality!)
- Step-Back Prompting
- Complexity Classification
- Reciprocal Rank Fusion

**Part 3 - Smart Routing** (🟠)
- 6 route types
- Tool selection (vector_db, sql, web)
- Full pipeline integration

**What You'll Build:**
- Optimize real queries
- See before/after comparisons
- Understand automatic strategy selection

**Key Takeaway:** Use `optimize_query_for_retrieval()` and let the system decide!

---

## 🚀 Quick Start

### Setup

1. **Clone the repo:**
```bash
git clone https://github.com/mahimaarora/retrieval-playground.git
cd retrieval-playground
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set environment variables:**
```bash
# Create .env file
GOOGLE_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_KEY=your_qdrant_key
```

4. **Download sample data:**
- Go to: [Research Papers Dataset](https://huggingface.co/datasets/mahimaarora025/research_papers/tree/main/sample_research_papers)
- Download any PDF (we recommend `cot_paper.pdf`)
- Save to: `retrieval_playground/data/sample_research_papers/`

5. **Launch Jupyter:**
```bash
jupyter notebook retrieval_playground/tutorial/
```

---

## 📊 What Makes These Notebooks Different?

### ✅ Beginner-Friendly Features

1. **Progressive Complexity**
   - 🟢 Green = Start here (simple)
   - 🟡 Yellow = Intermediate
   - 🟠 Orange = Advanced
   - 🔴 Red = Expert

2. **Visual Learning**
   - Diagrams before code
   - Before/after comparisons
   - Clear color coding

3. **Hands-On Experiments**
   - "Try It Yourself" sections
   - Interactive code cells
   - Immediate feedback

4. **Clear Use Cases**
   - "When to use" for each method
   - Real-world scenarios
   - Decision trees

5. **No Overwhelm**
   - No cell > 20 lines of code
   - One concept at a time
   - Can complete in < 30 mins each

---

## 🎓 Learning Outcomes

After completing both notebooks, you'll be able to:

### Chunking Skills
✅ Choose the right chunking strategy for your use case
✅ Understand trade-offs (speed vs quality)
✅ Implement any of the 4 strategies in < 5 minutes
✅ Know when to use multimodal chunking

### Query Optimization Skills
✅ Fix common query problems automatically
✅ Use RAG Fusion for +15-19% quality improvement
✅ Route queries to appropriate handlers
✅ Combine multiple techniques effectively

### Integration Skills
✅ Build complete RAG pipelines
✅ Make informed optimization decisions
✅ Measure and compare results
✅ Avoid common pitfalls

---

## 🆘 Troubleshooting

### Common Issues

**"Collection not found" error:**
```python
# Make sure you ran the chunking cell first
manager.create_chunks(...)  # This creates the collection
```

**"API key not found":**
```python
# Check your .env file exists and has correct keys
GOOGLE_API_KEY=your_key_here
```

**"Module not found":**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**Slow processing:**
- 🟢 Recursive: ~10 seconds
- 🟡 Contextual: ~30 seconds (LLM calls)
- 🟠 Parent-Child: ~15 seconds
- 🔴 Docling: ~2-3 minutes (multimodal extraction)

---

## 📈 Performance Expectations

### Chunking Impact
| Strategy | Speed | Quality | Storage | Best For |
|----------|-------|---------|---------|----------|
| 🟢 Recursive | ⚡⚡⚡ | ⭐⭐ | Low | Standard docs |
| 🟡 Contextual | ⚡⚡ | ⭐⭐⭐ | Medium | Technical docs |
| 🟠 Parent-Child | ⚡⚡ | ⭐⭐⭐⭐ | High | Production |
| 🔴 Docling | ⚡ | ⭐⭐⭐⭐ | Medium | Tables/Images |

### Query Optimization Impact
| Technique | Improvement | Cost | Best For |
|-----------|-------------|------|----------|
| Expansion | +10-15% | Low | Vague queries |
| Decomposition | +15-20% | Low | Multi-questions |
| Rewriting | +20-25% | Low | Follow-ups |
| Multi-Query | +15-19% | Medium | Important queries |
| Step-Back | +10-15% | Low | Technical queries |
| Full Pipeline | +25-35% | Medium | Production |

---

## 🎯 Next Steps

After completing the tutorials:

1. **Experiment with your data**
   - Upload your own documents
   - Test with real queries
   - Measure improvements

2. **Build a complete RAG system**
   - Combine chunking + query optimization
   - Add evaluation metrics
   - Deploy to production

3. **Advanced topics** (Future notebooks)
   - Evaluation frameworks
   - Hyperparameter tuning
   - Custom optimizations

---

## 💡 Pro Tips

### For Chunking
- 🚀 **Start simple:** Use Recursive for first iteration
- 📊 **Measure first:** Baseline before optimizing
- 💰 **Consider costs:** LLM calls add up
- 🎯 **Optimize selectively:** Not every doc needs advanced chunking

### For Query Optimization
- 🤖 **Use auto-optimization:** `optimize_query_for_retrieval()`
- 🔄 **Combine techniques:** Multi-query + routing + fusion
- 📈 **Track metrics:** Measure improvement over baseline
- ⚡ **Balance speed vs quality:** Not every query needs everything

### General
- 📚 **Read code comments:** They explain why, not just what
- 🧪 **Experiment freely:** Notebooks are safe to modify
- 🔍 **Compare results:** Side-by-side is most instructive
- 💬 **Ask questions:** Use GitHub Issues

---

## 📚 Additional Resources

- **Documentation:** `retrieval_playground/src/pre_retrieval/`
- **Tests:** `retrieval_playground/tests/`
- **Implementation Details:** See source code for advanced usage
- **Research Papers:** Citations in `PRE_RETRIEVAL_ANALYSIS.md`

---

## 🎉 Ready to Learn?

**Start with:** `1A_Pre_Chunking_Methods.ipynb`

**Time commitment:** ~45 minutes total  
**Prerequisites:** Basic Python  
**Outcome:** Build optimized RAG systems with confidence!

Happy learning! 🚀
