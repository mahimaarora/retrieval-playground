# 🧩 Retrieval Playground

A Python toolkit for RAG experimentation and evaluation.

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
- 🧪 **RAG Evaluation**: Performance benchmarking with RAGAS metrics  
- ⚙️ **Model Management**: Unified LLM and embedding model interfaces  

## ⚡ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/retrieval-playground.git
cd retrieval-playground
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -e .
```

### 4. Set Up Jupyter Kernel
```bash
python -m ipykernel install --user --name=venv --display-name "pydata_tutorial"
```

### 5. Launch Jupyter Notebook
```bash
jupyter notebook
```

### 6. Select the Correct Kernel
- Open any tutorial notebook
- Go to **Kernel** → **Change kernel** → **pydata_tutorial**

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

Explore RAG techniques through hands-on Jupyter notebooks:

- **1A_Pre_Chunking_Methods.ipynb** - Evaluate and compare different document chunking strategies using RAGAS metrics
- **1B_Pre_Query_Methods.ipynb** - Demonstrate query expansion, decomposition, rewriting, and self-querying techniques  
- **2_Mid_Retrieval_Methods.ipynb** - Explore various retrieval methods including MMR, hybrid search, and reranking
- **3_Post_Retrieval.ipynb** - Compare document chain methods for combining retrieved content into final answers

## 📄 License

MIT License
