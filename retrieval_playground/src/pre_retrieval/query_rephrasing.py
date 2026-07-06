"""
Query Rephrasing Module

Optimizes user queries for better retrieval through:
1. Expansion - Add synonyms and context
2. Decomposition - Break compound queries into atomic parts
3. Rewriting - Make context-dependent queries standalone
4. Multi-Query Generation - Generate query variants for RAG Fusion
5. Step-Back Prompting - Generate broader conceptual queries
6. Complexity Analysis - Classify query complexity for routing
"""

from langchain_core.prompts import PromptTemplate
from retrieval_playground.utils.model_manager import model_manager

# ============================================================================
# INITIALIZATION
# ============================================================================

# Global LLM instance
llm = model_manager.get_llm()


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

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

MULTI_QUERY_TEMPLATE = PromptTemplate(
    input_variables=["query", "num_variants"],
    template="""
Generate {num_variants} alternative phrasings of the query below.
Each variant should express the same intent using different wording, keywords, and perspectives.

Rules:
- Each variant must be standalone and retrieval-ready
- Use synonyms and related terms
- Vary the phrasing style (question, statement, keyword-based)
- Maintain the original specificity level

Return ONLY a valid Python list of {num_variants} strings.
Example format: ["variant 1", "variant 2", "variant 3"]

Query: {query}

Output:
"""
)

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

STEP_BACK_TEMPLATE = PromptTemplate(
    input_variables=["query"],
    template="""
Given this specific question, generate a broader, more general question that captures the underlying concept.

The broader question should:
- Help retrieve background knowledge needed to answer the specific question
- Be more conceptual and less specific
- Cover the domain/topic rather than the exact detail

Return ONLY the broader question, nothing else.

Specific Question: {query}

Broader Question:
"""
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _invoke_llm(template: PromptTemplate, **kwargs) -> str:
    """
    Invoke LLM with template and return response.

    Args:
        template: PromptTemplate instance
        **kwargs: Template variables

    Returns:
        LLM response string
    """
    prompt = template.format(**kwargs)
    return llm.invoke(prompt).content.strip()


def _parse_list_response(response: str, fallback: str) -> list[str]:
    """
    Parse LLM response into list of queries.

    Handles various formats:
    - Python list notation: ["query1", "query2"]
    - Numbered list: 1. query1\n2. query2
    - Bullet list: - query1\n- query2
    - Markdown code blocks: ```python\n["query1"]\n```

    Args:
        response: LLM response string
        fallback: Fallback query if parsing fails

    Returns:
        List of query strings
    """
    # Remove markdown code blocks if present
    cleaned_response = response.strip()
    if cleaned_response.startswith("```"):
        # Extract content between ``` markers
        lines = cleaned_response.split('\n')
        # Remove first line (```python or ```) and last line (```)
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned_response = '\n'.join(lines).strip()

    # Try to eval as Python list first
    try:
        result = eval(cleaned_response)
        if isinstance(result, list) and all(isinstance(item, str) for item in result):
            return result
    except:
        pass

    # Parse as line-based list
    lines = [line.lstrip('•-*1234567890. ').strip()
             for line in cleaned_response.split('\n')
             if line.strip() and not line.strip().startswith('[') and not line.strip().startswith(']')]

    # Filter out empty strings and non-query lines
    lines = [line for line in lines if line and len(line) > 3]

    return lines if lines else [fallback]


def _is_atomic_query(query: str) -> bool:
    """
    Validate if query is atomic (single intent).

    Simple heuristic:
    - No multiple "and" conjunctions
    - Max 1 question mark
    - Not too long (< 20 words)

    Args:
        query: Query string to validate

    Returns:
        True if atomic, False otherwise
    """
    return (
        query.count(" and ") <= 0 and
        query.count("?") <= 1 and
        len(query.split()) <= 20
    )


# ============================================================================
# CORE QUERY TRANSFORMATION FUNCTIONS
# ============================================================================

def expand_query(query: str, num_variants: int = 1) -> list[str]:
    """
    Expand a query using LLM.

    - num_variants=1: Single expansion (backward compatible)
    - num_variants>1: Multi-query generation for RAG Fusion

    Args:
        query: Original query
        num_variants: Number of query variants to generate

    Returns:
        List of expanded queries

    Example:
        >>> expand_query("What is RAG?")
        ["What is Retrieval-Augmented Generation?"]

        >>> expand_query("How does attention work?", num_variants=3)
        ["How does attention work?", "Explain attention mechanism", "What is attention in transformers?"]
    """
    if num_variants == 1:
        # Single query expansion (backward compatible)
        result = _invoke_llm(QUERY_EXPANSION_TEMPLATE, query=query)
        return [result]
    else:
        # Multi-query generation for RAG Fusion
        result = _invoke_llm(MULTI_QUERY_TEMPLATE, query=query, num_variants=num_variants)
        variants = _parse_list_response(result, query)

        # Ensure we have exactly num_variants
        if len(variants) < num_variants:
            # Add original query if we don't have enough
            variants.append(query)

        return variants[:num_variants]


def decompose_query(query: str, max_sub_queries: int = 5) -> list[str]:
    """
    Decompose a query into atomic sub-queries if it contains multiple intents.

    Args:
        query: Original query
        max_sub_queries: Maximum number of sub-queries to generate

    Returns:
        List of atomic sub-queries

    Example:
        >>> decompose_query("What is RAG and how does it improve LLM outputs?")
        ["What is RAG?", "How does RAG improve LLM outputs?"]
    """
    response = _invoke_llm(QUERY_DECOMPOSITION_TEMPLATE, query=query)
    sub_queries = _parse_list_response(response, query)

    # Validate sub-queries are atomic
    valid_sub_queries = [sq for sq in sub_queries if _is_atomic_query(sq)]

    return valid_sub_queries[:max_sub_queries]


def rewrite_query(query: str, previous_conversation_history: str = "") -> str:
    """
    Rewrite a query to be context-independent if it depends on prior context.

    Args:
        query: Original query
        previous_conversation_history: Previous conversation context

    Returns:
        Rewritten standalone query

    Example:
        >>> rewrite_query("How does it work?", "We were discussing transformers")
        "How do transformers work?"
    """
    return _invoke_llm(
        QUERY_REWRITING_TEMPLATE,
        query=query,
        previous_conversation_history=previous_conversation_history
    )


def step_back_query(query: str) -> tuple[str, str]:
    """
    Generate a broader conceptual query alongside the original.

    Useful for:
    - Ambiguous queries lacking context
    - Queries needing background knowledge
    - Multi-hop reasoning questions

    Args:
        query: Original specific query

    Returns:
        (broader_query, original_query) tuple

    Example:
        >>> step_back_query("What is the time complexity of BERT's self-attention?")
        ("What is computational complexity in transformer models?",
         "What is the time complexity of BERT's self-attention?")
    """
    broader_query = _invoke_llm(STEP_BACK_TEMPLATE, query=query)
    return broader_query, query


# ============================================================================
# COMPLEXITY ANALYSIS
# ============================================================================

# Moved to retrieval_playground.utils.query_classifier
from retrieval_playground.utils.query_classifier import classify_query_complexity


# ============================================================================
# RECIPROCAL RANK FUSION (For Multi-Query Results)
# ============================================================================

def reciprocal_rank_fusion(results_list: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF formula: score(d) = sum over all rankings R of: 1 / (k + rank_R(d))
    where k is a constant (typically 60)

    Args:
        results_list: List of result lists, each containing dicts with 'chunk_id' or 'id'
        k: RRF constant (default 60)

    Returns:
        Merged and re-ranked list of results

    Example:
        >>> results1 = [{"chunk_id": "doc1"}, {"chunk_id": "doc2"}]
        >>> results2 = [{"chunk_id": "doc2"}, {"chunk_id": "doc3"}]
        >>> fused = reciprocal_rank_fusion([results1, results2])
        >>> fused[0]["chunk_id"]
        "doc2"  # Appears in both lists, ranked first
    """
    fused_scores = {}
    doc_map = {}

    for results in results_list:
        for rank, doc in enumerate(results):
            # Get document ID
            doc_id = doc.get("chunk_id") or doc.get("id") or str(doc)

            # Track fused score
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_map[doc_id] = doc

            # RRF scoring
            fused_scores[doc_id] += 1 / (k + rank)

    # Sort by fused score (descending)
    ranked_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    # Return docs in fused rank order
    return [doc_map[doc_id] for doc_id, score in ranked_ids]


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def optimize_query_for_retrieval(
    query: str,
    context: str = "",
    auto_strategy: bool = True
) -> dict:
    """
    Main orchestration function for query optimization.

    Automatically selects and applies the best strategy based on query complexity.

    Args:
        query: Original user query
        context: Previous conversation context (optional)
        auto_strategy: Auto-select strategy based on complexity (default True)

    Returns:
        {
            "original_query": str,
            "processed_queries": list[str],
            "strategy": str,
            "complexity": dict,
            "metadata": dict
        }

    Example:
        >>> result = optimize_query_for_retrieval("Compare BERT and GPT")
        >>> result["strategy"]
        "multi_query"
        >>> len(result["processed_queries"])
        3
    """
    # Step 1: Rewrite if context-dependent
    if context:
        query = rewrite_query(query, context)

    # Step 2: Classify complexity
    complexity_analysis = classify_query_complexity(query)

    # Step 3: Apply appropriate strategy
    if auto_strategy:
        strategy = complexity_analysis["recommended_strategy"]
    else:
        strategy = "standard_rag"

    # Step 4: Process query based on strategy
    if strategy == "standard_rag":
        processed_queries = [expand_query(query, num_variants=1)[0]]

    elif strategy == "multi_query":
        processed_queries = expand_query(query, num_variants=3)

    elif strategy == "step_back":
        broader, specific = step_back_query(query)
        processed_queries = [broader, specific]

    elif strategy == "decompose":
        processed_queries = decompose_query(query)

    else:
        processed_queries = [query]

    return {
        "original_query": query,
        "processed_queries": processed_queries,
        "strategy": strategy,
        "complexity": complexity_analysis,
        "metadata": {
            "num_queries": len(processed_queries),
            "requires_fusion": strategy in ["multi_query", "decompose"],
            "context_used": bool(context)
        }
    }


# ============================================================================
# DEMO & EXAMPLES
# ============================================================================

def demo_query_rephrasing():
    """
    Demonstrate all query rephrasing techniques with workshop dataset examples.
    """
    print("=" * 80)
    print("QUERY REPHRASING DEMO - SciPy Workshop Dataset")
    print("=" * 80)

    # 1. Single Query Expansion
    print("\n🔍 1. SINGLE QUERY EXPANSION")
    print("-" * 50)
    query = "What is AL?"
    print(f"Original: {query}")
    result = expand_query(query, num_variants=1)
    print(f"Expanded: {result[0]}")

    # 2. Multi-Query Generation (RAG Fusion)
    print("\n🔍 2. MULTI-QUERY GENERATION (RAG Fusion)")
    print("-" * 50)
    query = "How do quantum graph neural networks improve molecular property prediction?"
    print(f"Original: {query}")
    variants = expand_query(query, num_variants=3)
    print("Variants:")
    for i, v in enumerate(variants, 1):
        print(f"  {i}. {v}")

    # 3. Query Decomposition
    print("\n📋 3. QUERY DECOMPOSITION")
    print("-" * 50)
    query = "What is AutoClimDS and how does it help with climate data analysis?"
    print(f"Original: {query}")
    sub_queries = decompose_query(query)
    print("Sub-queries:")
    for i, sq in enumerate(sub_queries, 1):
        print(f"  {i}. {sq}")

    # 4. Step-Back Prompting
    print("\n🔙 4. STEP-BACK PROMPTING")
    print("-" * 50)
    query = "What specific CUDA optimization techniques improve GPU performance in transformer models?"
    print(f"Specific: {query}")
    broader, _ = step_back_query(query)
    print(f"Broader: {broader}")

    # 5. Complexity Classification
    print("\n📊 5. COMPLEXITY CLASSIFICATION")
    print("-" * 50)
    queries = [
        "What is Agent Laboratory?",
        "Compare the differences between PyTorch and JAX frameworks and explain which is better for scientific computing",
        "Explain how quantum graph neural networks improve molecular property prediction and why they outperform classical graph neural networks"
    ]
    for query in queries:
        result = classify_query_complexity(query)
        print(f"Query: {query}")
        print(f"  Complexity: {result['complexity']} (score: {result['score']})")
        print(f"  Strategy: {result['recommended_strategy']}")
        print()

    # 6. Full Orchestration
    print("\n🎯 6. FULL ORCHESTRATION")
    print("-" * 50)
    query = "Compare the performance differences between Pandas, Polars, and Dask dataframe libraries and explain which is best for large-scale scientific data processing"
    print(f"Query: {query}")
    result = optimize_query_for_retrieval(query)
    print(f"Strategy: {result['strategy']}")
    print(f"Complexity: {result['complexity']['complexity']}")
    print(f"Signals: {result['complexity']['signals']}")
    print(f"Processed Queries ({result['metadata']['num_queries']}):")
    for i, pq in enumerate(result['processed_queries'], 1):
        print(f"  {i}. {pq}")

    # 7. Context-Aware Rewriting
    print("\n📝 7. CONTEXT-AWARE REWRITING")
    print("-" * 50)
    previous_context = "We were discussing AI agents for scientific research and their ability to automate experiments."
    query = "How effective are they in practice?"
    print(f"Context: {previous_context}")
    print(f"Original: {query}")
    rewritten = rewrite_query(query, previous_context)
    print(f"Rewritten: {rewritten}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demo_query_rephrasing()
