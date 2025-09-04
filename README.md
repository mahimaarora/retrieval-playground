# Retrieval Playground

A Python toolkit for retrieval-augmented generation (RAG) experimentation and evaluation.

## Features

- **Pre-retrieval Processing**: Chunking strategies, query rephrasing, semantic routing
- **Document Processing**: PDF text extraction and synthetic test data generation  
- **Model Management**: Unified LLM and embedding model interfaces
- **Evaluation Tools**: RAG performance benchmarking with RAGAS metrics

## Installation

```bash
git clone https://github.com/yourusername/retrieval-playground.git
cd retrieval-playground
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

## Quick Start

```python
from retrieval_playground import ModelManager
from retrieval_playground.src.pre_retrieval.chunking_strategies import PreRetrievalChunking
from retrieval_playground.src.pre_retrieval.query_rephrasing import expand_query

# Initialize models and chunking
model_manager = ModelManager()
llm = model_manager.get_llm()
chunker = PreRetrievalChunking()

# Process documents and queries
chunks = chunker.chunk_documents(documents, strategy="docling")
enhanced_query = expand_query("What is machine learning?")
```

### Generate Test Data

```bash
rp-generate-test-data
```

## Configuration

Set your Google API key:
```bash
export GOOGLE_API_KEY="your-google-api-key"
```

Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

## Project Structure

```
retrieval_playground/
├── src/
│   ├── pre_retrieval/          # Pre-processing techniques
│   ├── baseline_rag.py         # Basic RAG implementation
│   └── evaluation.py           # Performance evaluation
├── data/                       # Sample documents and results
├── tests/                      # Test data generation
└── utils/                      # Shared utilities
```

## License

MIT License