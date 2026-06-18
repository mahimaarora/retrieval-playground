"""
Routing Module

Routes queries to appropriate handlers using semantic similarity.

Features:
1. Semantic Routing - Route based on query intent
2. Complexity-Based Routing - Simple vs complex query handling
3. Tool Selection - Choose appropriate retrieval method
"""

from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.routers import SemanticRouter
from retrieval_playground.utils import constants

# ============================================================================
# CONFIGURATION
# ============================================================================

ROUTE_SIMILARITY_THRESHOLD = 0.7
ROUTE_CONFIDENCE_PRECISION = 3


# ============================================================================
# ROUTE DEFINITIONS
# ============================================================================

greetings_route = Route(
    name="greetings",
    utterances=[
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "how are you", "how are you doing", "how's it going", "what's up",
        "how do you do", "what's new", "what's going on", "what's happening",
        "thank you", "thanks", "thank you very much", "appreciate it",
        "goodbye", "bye", "see you later", "see you soon", "farewell",
        "nice to meet you", "pleasure to meet you", "good to see you",
        "how can I help", "what can you do", "what are your capabilities"
    ],
)

factual_qa_route = Route(
    name="factual_qa",
    utterances=[
        "what is", "what are", "who is", "when was", "where is",
        "define", "definition of", "meaning of",
        "what year", "what date", "how many",
        "list the", "name the", "identify",
        "what does", "which is", "tell me about"
    ],
)

analytical_qa_route = Route(
    name="analytical_qa",
    utterances=[
        "explain", "why", "how does", "analyze",
        "what causes", "what led to", "reasoning behind",
        "implications of", "impact of", "effect of",
        "evaluate", "assess", "discuss",
        "describe how", "elaborate on"
    ],
)

comparison_route = Route(
    name="comparison",
    utterances=[
        "difference between", "compare", "versus", "vs",
        "better than", "worse than", "similar to",
        "contrast", "comparison of", "how does X differ from Y",
        "advantages of", "disadvantages of",
        "differences", "similarities"
    ],
)

definition_route = Route(
    name="definition",
    utterances=[
        "what is a", "what does X mean", "define",
        "definition of", "meaning of", "term",
        "concept of", "what do you mean by",
        "explain the term", "what is meant by",
        "terminology", "glossary"
    ],
)

procedural_route = Route(
    name="procedural",
    utterances=[
        "how to", "how do I", "how can I",
        "steps to", "process for", "procedure for",
        "guide to", "tutorial on", "instructions for",
        "way to", "method to", "approach to",
        "show me how", "walk me through"
    ],
)

# All routes
routes = [
    greetings_route,
    factual_qa_route,
    analytical_qa_route,
    comparison_route,
    definition_route,
    procedural_route,
]


# ============================================================================
# SEMANTIC ROUTER INITIALIZATION
# ============================================================================

rl = SemanticRouter(
    encoder=HuggingFaceEncoder(
        name=constants.DEFAULT_EMBEDDING_MODEL,
        score_threshold=ROUTE_SIMILARITY_THRESHOLD,
        trust_remote_code=True
    ),
    routes=routes,
    auto_sync="local"
)


# ============================================================================
# ROUTE METADATA
# ============================================================================

ROUTE_METADATA = {
    "greetings": {
        "requires_retrieval": False,
        "retrieval_method": None,
        "tool": "llm_direct",
        "complexity": "simple",
        "use_reranking": False,
        "description": "Casual conversation and greetings"
    },
    "factual_qa": {
        "requires_retrieval": True,
        "retrieval_method": "hybrid_search",
        "tool": "vector_db",
        "complexity": "simple",
        "use_reranking": False,
        "description": "Simple fact lookup questions"
    },
    "analytical_qa": {
        "requires_retrieval": True,
        "retrieval_method": "dense_search",
        "tool": "vector_db",
        "complexity": "moderate",
        "use_reranking": True,
        "description": "Questions requiring reasoning and analysis"
    },
    "comparison": {
        "requires_retrieval": True,
        "retrieval_method": "multi_query",
        "tool": "vector_db",
        "complexity": "moderate",
        "use_reranking": True,
        "description": "Comparison between entities or concepts"
    },
    "definition": {
        "requires_retrieval": True,
        "retrieval_method": "hybrid_search",
        "tool": "vector_db",
        "complexity": "simple",
        "use_reranking": False,
        "description": "Definition and terminology questions"
    },
    "procedural": {
        "requires_retrieval": True,
        "retrieval_method": "dense_search",
        "tool": "vector_db",
        "complexity": "moderate",
        "use_reranking": True,
        "description": "How-to and step-by-step questions"
    },
}


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def semantic_layer(
    query: str,
    similarity_threshold: float = ROUTE_SIMILARITY_THRESHOLD,
    return_metadata: bool = False
):
    """
    Apply semantic routing to determine query handling strategy.

    Args:
        query: Input query string
        similarity_threshold: Minimum similarity score for route activation
        return_metadata: If True, return full routing decision dict

    Returns:
        If return_metadata=False: enhanced query string (backward compatible)
        If return_metadata=True: routing decision dict
    """
    try:
        route = rl(query)
        similarity_score = round(route.similarity_score, ROUTE_CONFIDENCE_PRECISION) if route.similarity_score else 0.0

        # Get route decision
        route_decision = _get_route_decision(route.name, similarity_score, query)

        if return_metadata:
            return route_decision
        else:
            # Backward compatible: return enhanced query string
            enhanced_query = f"{query} [SYSTEM_NOTE: {route_decision['system_note']}]"
            return enhanced_query

    except Exception as e:
        # Fallback
        fallback_decision = {
            "route_name": "default",
            "confidence": 0.0,
            "requires_retrieval": True,
            "retrieval_method": "hybrid_search",
            "tool": "vector_db",
            "complexity": "moderate",
            "use_reranking": False,
            "system_note": f"ROUTING: Error ({str(e)}). Using default retrieval."
        }

        if return_metadata:
            return fallback_decision
        else:
            return f"{query} [SYSTEM_NOTE: {fallback_decision['system_note']}]"


def _get_route_decision(route_name: str, similarity_score: float, query: str) -> dict:
    """
    Get routing decision based on route name and confidence.

    Args:
        route_name: Name of the detected route
        similarity_score: Confidence score
        query: Original query

    Returns:
        Routing decision dictionary
    """
    # Get metadata for this route
    metadata = ROUTE_METADATA.get(route_name, {
        "requires_retrieval": True,
        "retrieval_method": "hybrid_search",
        "tool": "vector_db",
        "complexity": "moderate",
        "use_reranking": False,
        "description": "Unknown route"
    })

    decision = {
        "route_name": route_name,
        "confidence": similarity_score,
        "requires_retrieval": metadata["requires_retrieval"],
        "retrieval_method": metadata["retrieval_method"],
        "tool": metadata["tool"],
        "complexity": metadata["complexity"],
        "use_reranking": metadata["use_reranking"],
        "system_note": _format_system_note(route_name, similarity_score, metadata)
    }

    return decision


def _format_system_note(route_name: str, confidence: float, metadata: dict) -> str:
    """
    Format system note for backward compatibility.

    Args:
        route_name: Route name
        confidence: Confidence score
        metadata: Route metadata

    Returns:
        Formatted system note string
    """
    if not metadata["requires_retrieval"]:
        return f"ROUTING: {route_name} (no retrieval needed) [Confidence: {confidence}]"
    else:
        method = metadata["retrieval_method"]
        return f"ROUTING: {route_name} → {method} [Confidence: {confidence}]"


def route_with_complexity_analysis(query: str) -> dict:
    """
    Route query considering both semantic similarity and complexity.

    Combines semantic routing with complexity analysis for better decisions.

    Args:
        query: Input query

    Returns:
        {
            "route": dict (from semantic_layer),
            "complexity": dict (from classify_query_complexity),
            "final_strategy": str,
            "final_retrieval_method": str
        }
    """
    # Import here to avoid circular dependency
    from .query_rephrasing import classify_query_complexity

    # Step 1: Semantic routing
    route_decision = semantic_layer(query, return_metadata=True)

    # Step 2: Complexity analysis
    complexity_analysis = classify_query_complexity(query)

    # Step 3: Decide final strategy
    # If complexity says "complex" but route says "simple", upgrade
    if complexity_analysis["complexity"] == "complex" and route_decision["complexity"] == "simple":
        final_strategy = complexity_analysis["recommended_strategy"]
        final_retrieval_method = "multi_query" if final_strategy == "multi_query" else "dense_search"
    else:
        # Use route decision
        final_strategy = complexity_analysis["recommended_strategy"]
        final_retrieval_method = route_decision["retrieval_method"] or "dense_search"

    return {
        "route": route_decision,
        "complexity": complexity_analysis,
        "final_strategy": final_strategy,
        "final_retrieval_method": final_retrieval_method,
        "metadata": {
            "route_name": route_decision["route_name"],
            "confidence": route_decision["confidence"],
            "complexity_score": complexity_analysis["score"],
            "signals": complexity_analysis["signals"]
        }
    }


def select_retrieval_tool(
    query: str,
    route_decision: dict,
    available_tools: list = None
) -> str:
    """
    Select appropriate retrieval tool based on query and route.

    Available tools:
    - vector_db: Semantic vector search (default)
    - bm25: Keyword/sparse search
    - hybrid: BM25 + vector fusion
    - sql: Structured data queries
    - web: Web search for recent events

    Args:
        query: User query
        route_decision: Output from route_with_complexity_analysis()
        available_tools: List of available tools (default: all)

    Returns:
        Tool name to use
    """
    if available_tools is None:
        available_tools = ["vector_db", "bm25", "hybrid", "sql", "web"]

    # Check for SQL indicators
    sql_indicators = ["count", "total", "average", "sum", "how many", "statistics"]
    if any(indicator in query.lower() for indicator in sql_indicators):
        if "sql" in available_tools:
            return "sql"

    # Check for recency indicators (need web search)
    recency_indicators = ["latest", "recent", "current", "today", "2026", "now", "this year"]
    if any(indicator in query.lower() for indicator in recency_indicators):
        if "web" in available_tools:
            return "web"

    # Otherwise use route recommendation
    method = route_decision.get("final_retrieval_method", "dense_search")

    if method == "hybrid_search" and "hybrid" in available_tools:
        return "hybrid"
    elif method == "sparse_search" and "bm25" in available_tools:
        return "bm25"
    else:
        return "vector_db"  # Default


# ============================================================================
# UTILITIES
# ============================================================================

def get_route_info():
    """
    Get information about available routes and their configurations.

    Returns:
        Dict containing route information and statistics
    """
    route_info = {
        "total_routes": len(routes),
        "routes": {},
        "embedding_model": constants.DEFAULT_EMBEDDING_MODEL,
        "default_threshold": ROUTE_SIMILARITY_THRESHOLD
    }

    for route in routes:
        route_info["routes"][route.name] = {
            "name": route.name,
            "utterances_count": len(route.utterances),
            "metadata": ROUTE_METADATA.get(route.name, {})
        }

    return route_info


# ============================================================================
# DEMO
# ============================================================================

def demo_routing():
    """
    Demonstrate routing functionality with examples.
    """
    print("=" * 80)
    print("ROUTING DEMO")
    print("=" * 80)

    # Test queries for each route
    test_queries = [
        ("Hi there!", "greetings"),
        ("What is RAG?", "factual_qa"),
        ("Explain how transformers work", "analytical_qa"),
        ("Compare BERT and GPT", "comparison"),
        ("Define attention mechanism", "definition"),
        ("How to implement RAG?", "procedural"),
    ]

    print(f"\nConfigured Routes: {len(routes)}")
    print(f"Similarity Threshold: {ROUTE_SIMILARITY_THRESHOLD}")
    print()

    for query, expected_route in test_queries:
        print(f"\nQuery: {query}")
        print(f"Expected Route: {expected_route}")

        # Test semantic_layer with metadata
        result = semantic_layer(query, return_metadata=True)
        print(f"Detected Route: {result['route_name']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Retrieval Method: {result['retrieval_method']}")
        print(f"Requires Retrieval: {result['requires_retrieval']}")
        print("-" * 80)

    # Test complexity-based routing
    print("\n" + "=" * 80)
    print("COMPLEXITY-BASED ROUTING")
    print("=" * 80)

    complex_queries = [
        "What is RAG?",
        "Compare BERT and GPT-3 performance on question answering tasks"
    ]

    for query in complex_queries:
        print(f"\nQuery: {query}")
        result = route_with_complexity_analysis(query)
        print(f"Route: {result['route']['route_name']}")
        print(f"Complexity: {result['complexity']['complexity']} (score: {result['complexity']['score']})")
        print(f"Final Strategy: {result['final_strategy']}")
        print(f"Final Retrieval Method: {result['final_retrieval_method']}")

    # Test tool selection
    print("\n" + "=" * 80)
    print("TOOL SELECTION")
    print("=" * 80)

    tool_queries = [
        "What is RAG?",
        "How many papers discuss transformers?",
        "What are the latest developments in LLMs in 2026?"
    ]

    for query in tool_queries:
        route_decision = route_with_complexity_analysis(query)
        tool = select_retrieval_tool(query, route_decision)
        print(f"\nQuery: {query}")
        print(f"Selected Tool: {tool}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demo_routing()
