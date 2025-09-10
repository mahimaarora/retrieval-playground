from langchain.prompts import PromptTemplate
from retrieval_playground.utils.model_manager import model_manager

# Global LLM instance
llm = model_manager.get_llm()

# Query expansion prompt template
QUERY_EXPANSION_TEMPLATE = PromptTemplate(
    input_variables=["query"],
        template="""
Given the query below, decide whether it needs expansion.  

Expand the query if any of the following apply:  
- It contains abbreviations or acronyms → replace them with their full forms.  
- It is too broad or vague → add minimal context to make it retrieval-ready.  
- It lacks domain-specific terms that are typically associated with the topic → enrich with relevant context.  
- It is just a direct phrase or incomplete question → reframe it into a clear query/question suitable for retrieval.  

If none of these apply, return the query exactly as it is.  

The output should be a natural search query suitable for retrieval.  
Do not include explanations, just return the final query text.  

Query: {query}  
Output:
"""
)

# Query decomposition prompt template
QUERY_DECOMPOSITION_TEMPLATE = PromptTemplate(
    input_variables=["query"],
    template="""
Given the query below, check if it contains multiple intents or compound questions.  
- If it does, break it down into smaller, atomic sub-queries.  
- Each sub-query must be an independent, standalone query that can be retrieved without relying on the others.  
- If not, return the query inside a single-item list.  

Return only a valid Python list of sub-queries.  

Query: {query}  
Output:
"""
)

# Query rewriting prompt template
QUERY_REWRITING_TEMPLATE = PromptTemplate(
    input_variables=["query", "previous_conversation_history"],
    template="""
Given the current query and the previous conversation history:  
- If the query depends on prior context (e.g., pronouns, references, incomplete information), rewrite it into a clear, standalone query suitable for retrieval.  
- If it does not depend on prior context, return the query unchanged.  

Return only the final query text, without explanation or formatting.  

Query: {query}  
Previous conversation history: {previous_conversation_history}  
Output:
"""
)

# Self-querying prompt template
SELF_QUERYING_TEMPLATE = PromptTemplate(
    input_variables=["query"],
    template="""
Transform the input into a set of optimal search queries for retrieval.  
- Queries should be clear, focused, and aligned with the user’s intent.  
- Each query must be independent and standalone.  

Return only a valid Python list of search queries.  

Input: {query}  
Output:
"""
)

def _parse_list_response(response: str, fallback: str) -> list[str]:
    """Parse LLM response into list of queries."""
    lines = [line.lstrip('•-*1234567890. ').strip() 
             for line in response.split('\n') if line.strip()]
    return lines if lines else [fallback]

def _invoke_llm(template, **kwargs) -> str:
    """Invoke LLM with template and return response."""
    prompt = template.format(**kwargs)
    return llm.invoke(prompt).content.strip()

def expand_query(query: str) -> str:
    """Expand a query using LLM if beneficial."""
    return _invoke_llm(QUERY_EXPANSION_TEMPLATE, query=query)

def decompose_query(query: str) -> list[str]:
    """Decompose a query into atomic sub-queries if it contains multiple intents."""
    response = _invoke_llm(QUERY_DECOMPOSITION_TEMPLATE, query=query)
    return _parse_list_response(response, query)

def rewrite_query(query: str, previous_conversation_history: str = "") -> str:
    """Rewrite a query to be context-independent if it depends on prior context."""
    return _invoke_llm(QUERY_REWRITING_TEMPLATE, 
                      query=query, 
                      previous_conversation_history=previous_conversation_history)

def self_query(query: str) -> list[str]:
    """Transform user input into optimal search queries for retrieval."""
    response = _invoke_llm(SELF_QUERYING_TEMPLATE, query=query)
    return _parse_list_response(response, query)


# Example queries aligned with research papers theme
QUERY_EXAMPLES = {
    "expansion": [
        {"query": "Generative AI transformer evaluation benchmarks"},
        {"query": "Retrieved capabilities of LLMs and FLOPs scaling"},
        {"query": "AI models"},
        {"query": "tell me about computer vision"},
        {"query": "Physics-informed neural network for fatigue life prediction"},
    ],
    
    "decomposition": [
        {"query":"What are neural networks and how do they work in computer vision applications?"},
        {"query":"Explain machine learning algorithms, their types, and performance evaluation metrics"},
        {"query":"What is generative AI, what are its applications, and what are the ethical considerations?"},
        {"query":"How do statistical models work and what are their advantages over machine learning approaches?"},
        {"query":"What are the latest advances in computer vision and how do they apply to remote sensing?"}
    ],
    
    "rewriting": [
        {"query":"How does it work?", "previous_conversation_history":"User asked about counterfactual generation in machine learning. Assistant explained that it's a method for creating hypothetical scenarios to understand causal relationships in data."},
        {"query":"What are the main advantages?", "previous_conversation_history":"User inquired about Riemannian manifolds for change point detection. Assistant described how these mathematical structures can capture complex data geometries for robust statistical analysis."},
        {"query":"Can you explain the methodology?", "previous_conversation_history":"User asked about publication bias in meta-analysis research. Assistant mentioned that Copas-Jackson bounds are statistical techniques used to assess and correct for selective reporting in academic studies."},
        {"query":"What about the computational complexity?", "previous_conversation_history":"User was learning about state space models in generative AI. Assistant explained that these models represent sequential data through hidden states and observable outputs."},
        {"query":"How does this compare to traditional approaches?", "previous_conversation_history":"User asked about annotation-free segmentation methods for remote sensing. Assistant described how these computer vision techniques can identify objects without requiring manually labeled training data."}
    ],
    
    "self_querying": [
        {"query":"I need to understand the current state of research in computer vision, particularly focusing on segmentation techniques for remote sensing applications. What are the key papers and methodologies?"},
        {"query":"Please explain the mathematical foundations of manifold learning and how it applies to change point detection. I'm particularly interested in robust estimation methods."},
        {"query":"Help me understand generative AI models, their computational complexity, and the satisfiability problems in state space models. Focus on recent advances."},
        {"query":"I want to learn about publication bias in statistical research. Explain the Copas-Jackson bounds and similar methods for addressing selection bias in meta-analysis."},
        {"query":"Provide an overview of counterfactual generation methods in machine learning, emphasizing model-agnostic approaches and causal constraints."}
    ]
}


def demo_query_processing():
    """
    Demonstrate all query processing techniques using the provided examples.
    Prints results in a structured, easy-to-understand format.
    """
    print("=" * 80)
    print("QUERY PROCESSING TECHNIQUES DEMO")
    print("=" * 80)
    
    # Query Expansion Demo
    print("\n🔍 QUERY EXPANSION")
    print("-" * 50)
    print("Purpose: Add synonyms, related terms, or contextual entities")
    print()
    
    for i, example in enumerate(QUERY_EXAMPLES["expansion"], 1):
        query = example["query"]
        print(f"{i}. Original: {query}")
        try:
            expanded = expand_query(query)
            print(f"   Expanded: {expanded}")
        except Exception as e:
            print(f"   Error: {e}")
        print()
    
    # Query Decomposition Demo
    print("\n📋 QUERY DECOMPOSITION")
    print("-" * 50)
    print("Purpose: Break compound queries into atomic sub-queries")
    print()
    
    for i, example in enumerate(QUERY_EXAMPLES["decomposition"], 1):
        query = example["query"]
        print(f"{i}. Original: {query}")
        try:
            sub_queries = decompose_query(query)
            print(f"   Sub-queries:")
            for j, sub_query in enumerate(sub_queries, 1):
                print(f"{sub_query}")
        except Exception as e:
            print(f"   Error: {e}")
        print()
    
    # Query Rewriting Demo
    print("\n✏️  QUERY REWRITING")
    print("-" * 50)
    print("Purpose: Make context-dependent queries standalone")
    print()
    
    for i, example in enumerate(QUERY_EXAMPLES["rewriting"], 1):
        query = example["query"]
        context = example["previous_conversation_history"]
        print(f"{i}. Context-dependent: {query}")
        print(f"   Context: {context}")
        try:
            rewritten = rewrite_query(query, context)
            print(f"   Standalone: {rewritten}")
        except Exception as e:
            print(f"   Error: {e}")
        print()
    
    # Self-Querying Demo
    print("\n🎯 SELF-QUERYING")
    print("-" * 50)
    print("Purpose: Transform complex input into optimal search queries")
    print()
    
    for i, example in enumerate(QUERY_EXAMPLES["self_querying"], 1):
        query = example["query"]
        print(f"{i}. Complex Input: {query}")
        try:
            search_queries = self_query(query)
            print(f"   Optimal Search Queries:")
            for j, search_query in enumerate(search_queries, 1):
                print(f"{search_query}")
        except Exception as e:
            print(f"   Error: {e}")
        print()
    
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demo_query_processing()


