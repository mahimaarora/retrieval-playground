# 🧩 Retrieval Playground

A Python toolkit for RAG experimentation and evaluation.

> ⚠️ **Work in Progress**: This repository is actively being developed for workshops and tutorials. Some features may be incomplete or subject to change.

## ✨ Features

### Pre-Retrieval Strategies

- **Chunking**: Baseline, recursive character, unstructured title-based, and Docling hybrid chunking  
- **Query Enhancement**: Query expansion, decomposition, rewriting, and self-querying  
- **Semantic Routing**: Route queries to appropriate knowledge domains with confidence scoring

### Mid-Retrieval Strategies

- **Basic Similarity Search**: Standard semantic search with vector databases  
- **MMR (Maximal Marginal Relevance)**: Balance relevance and diversity in results  
- **Score Thresholding**: Quality-based filtering of retrieval results  
- **Metadata Filtering**: Context-aware search with document attributes  
- **Reranking**: Cross-encoder models to reorder results for higher precision  
- **Hybrid Retrieval**: Combine BM25 keyword search with dense semantic search

### Post-Retrieval Strategies

- **Stuff Documents**: Simple concatenation of all retrieved documents  
- **Refine Chain**: Iterative refinement of answers across documents  
- **Map-Rerank**: Score and rank answers from individual documents  
- **Map-Reduce**: Summarize documents first, then combine for final answer

### Evaluation & Management

- **RAG Evaluation**: Performance benchmarking with RAGAS metrics  
- **Model Management**: Unified LLM and embedding model interfaces

## Prerequisites

1. Create a Google account and generate an API key to access Gemini’s free tier:
  - Visit [Gemini API Docs](https://ai.google.dev/gemini-api/docs)
    - In the top-right corner, click Get API Key. 
    - Follow the instructions, copy your API key, and save it in a .env file under the variable GOOGLE_API_KEY.
2. [Optional] Install your preferred IDE (e.g., Visual Studio, PyCharm), or use the terminal as an alternative.### 3. Install Python ≥ 3.12
3. This project requires Python 3.12 or higher. Follow the official installation guide for your operating system:
  - **Windows:** [Python for Windows](https://docs.python.org/3/using/windows.html)  
  - **macOS:** [Python for macOS](https://docs.python.org/3/using/mac.html)  
  - **Linux:** [Python for Linux/UNIX](https://docs.python.org/3/using/unix.html)
   After installing, verify with:

## 🚀 Quick Start

### For Workshop Participants

**Follow the setup guide for your operating system:**

**[Setup Guides](setup-guides/)** - Choose your OS:

- [Mac Setup](setup-guides/SETUP_MAC.md)
- [Windows](setup-guides/SETUP_WINDOWS.md) 
- [Linux Setup](setup-guides/SETUP_LINUX.md)

The setup guides cover:

1. Installing Docker
2. Getting the workshop files
3. Configuring API keys
4. Starting the environment
5. Verifying your setup

**Time needed:** 20-30 minutes for first-time setup

---

## 📋 Prerequisites

1. Create a Google account and generate an API key to access Gemini's free tier:
  - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
    - Click "Get API Key" or "Create API Key"
    - Follow the instructions, copy your API key, and save it in a .env file under the variable GOOGLE_API_KEY
2. [Optional] For Qdrant vector database:
  - **During workshop:** Use instructor-provided credentials
    - **For personal use:** Create free account at [Qdrant Cloud](https://cloud.qdrant.io/)

---

## Alternative: Manual Installation (Advanced Users)

If you prefer not to use Docker:

### 1. Clone the Repository

```bash
git clone https://github.com/mahimaarora/retrieval-playground.git
cd retrieval-playground
```

### 2. Set Up Virtual Environment

```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -e .
```

### 4. Set Up Jupyter Kernel

```bash
python -m ipykernel install --user --name=venv --display-name "scipy_tutorial"
```

### 5. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 6. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 7. Select the Correct Kernel

- Open any tutorial notebook
- Go to **Kernel** → **Change kernel** → **scipy_tutorial**

## 🚀 Quick Start

```python
from retrieval_playground import ModelManager
from retrieval_playground.src.pre_retrieval.chunking_strategies import PreRetrievalChunking

# Initialize and use
model_manager = ModelManager()
chunker = PreRetrievalChunking()
chunks = chunker.chunk_documents(documents, strategy="docling")
```

## 📓 Interactive Notebooks

Explore RAG techniques through hands-on Jupyter notebooks located in `retrieval_playground/tutorial/`:

- **1A_Pre_Chunking_Methods.ipynb** - Evaluate and compare different document chunking strategies using RAGAS metrics
- **1B_Pre_Query_Methods.ipynb** - Demonstrate query expansion, decomposition, rewriting, and self-querying techniques  
- **2_Mid_Retrieval_Methods.ipynb** - Explore various retrieval methods including MMR, hybrid search, and reranking
- **3_Post_Retrieval.ipynb** - Compare document chain methods for combining retrieved content into final answers

## 📄 License

MIT License