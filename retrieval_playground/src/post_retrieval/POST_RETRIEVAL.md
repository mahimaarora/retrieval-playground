# Post-Retrieval Processing

This module processes and combines retrieved documents **after** retrieval to generate optimal responses in RAG systems.

## 📁 Contents

### 📚 **Document Chain Methods** (`3_Post_Retrieval.ipynb`)
Four powerful approaches for combining retrieved documents:

#### 📄 **Stuff Documents Chain**
- **Simple concatenation**: Combines all documents into single prompt
- **Best for**: Small document sets, straightforward queries
- **Advantages**: Fast, preserves all context
- **Limitations**: Token limits with large documents

#### 🔄 **Refine Documents Chain** 
- **Iterative refinement**: Progressively improves answers document by document
- **Best for**: Complex queries requiring nuanced answers
- **Advantages**: Handles large document sets, builds comprehensive responses
- **Process**: Initial answer → refine with each subsequent document

#### 📊 **Map-Rerank Chain**
- **Score and rank**: Generates individual answers, then selects the best
- **Best for**: Questions with clear single best answers
- **Advantages**: Identifies highest quality responses
- **Process**: Generate answers for each document → score → select top answer

#### 🔀 **Map-Reduce Chain**
- **Summarize then combine**: Summarizes documents first, then combines summaries
- **Best for**: Large document collections, synthesis tasks
- **Advantages**: Scales well, reduces token usage
- **Process**: Summarize each document → combine all summaries → final answer

## 🚀 Quick Start

### Interactive Notebook
- `retrieval_playground/tutorial/3_Post_Retrieval.ipynb` - Complete tutorial with both LangChain and LangGraph implementations

### Usage Examples

#### Traditional LangChain Approach
```python
from langchain.chains import StuffDocumentsChain, RefineDocumentsChain
from langchain.chains import MapRerankDocumentsChain, MapReduceDocumentsChain
from langchain.prompts import PromptTemplate

# Stuff Chain - Simple concatenation
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="docs"
)

# Refine Chain - Iterative improvement  
refine_chain = RefineDocumentsChain(
    initial_llm_chain=initial_chain,
    refine_llm_chain=refine_chain,
    document_variable_name="docs",
    initial_response_name="existing_answer"
)

# Map-Rerank Chain - Score and select best
map_rerank_chain = MapRerankDocumentsChain(
    llm_chain=llm_chain,
    rank_key="score",
    answer_key="answer"
)

# Map-Reduce Chain - Summarize then combine
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_chain,
    document_variable_name="docs"
)
```

## 🎯 Method Selection Guide

### When to Use Each Method

| Method | Best For | Document Count | Query Type | Performance |
|--------|----------|---------------|------------|-------------|
| **Stuff** | Simple Q&A | Small (1-5) | Direct questions | ⚡ Fastest |
| **Refine** | Complex analysis | Medium (5-15) | Multi-faceted queries | 🔄 Thorough |
| **Map-Rerank** | Single best answer | Any | Factual questions | 🎯 Precise |
| **Map-Reduce** | Synthesis tasks | Large (15+) | Summarization | 📊 Scalable |

### Implementation Approaches

#### ⚡ **Traditional LangChain**
- **Pros**: Mature, well-documented, extensive ecosystem
- **Cons**: Less flexible, harder to customize
- **Best for**: Standard use cases, rapid prototyping

#### 🔧 **Modern LangGraph** 
- **Pros**: Highly customizable, better control flow, modern architecture
- **Cons**: Newer, smaller ecosystem
- **Best for**: Complex workflows, custom logic, production systems

## 🔧 Configuration Options

### Chain Parameters
- **Temperature**: Control response creativity (0.0-1.0)
- **Max tokens**: Limit response length
- **Document separator**: How to join documents (stuff method)
- **Refinement iterations**: Number of refine steps (refine method)

### Prompt Templates
- **Question prompts**: How to frame questions to documents
- **Refinement prompts**: How to improve existing answers
- **Scoring prompts**: Criteria for ranking answers (map-rerank)
- **Summary prompts**: How to summarize individual documents (map-reduce)

## 📈 Performance Optimization

Post-retrieval processing improves RAG by:
- **Context utilization** → Optimal document combination strategies
- **Response quality** → Method-specific answer generation
- **Scalability** → Handle varying document counts efficiently
- **Flexibility** → Choose approach based on query complexity

## 🛠️ Advanced Features

### Custom Implementations
- **Hybrid chains**: Combine multiple methods
- **Conditional routing**: Choose method based on query type
- **Streaming responses**: Real-time answer generation
- **Error handling**: Graceful fallbacks for processing failures

### Integration Points
- **Pre-retrieval**: Works with any chunking strategy
- **Mid-retrieval**: Compatible with reranked results
- **Evaluation**: Built-in metrics for chain comparison
