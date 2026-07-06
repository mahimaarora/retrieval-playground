"""
Routing Module

Routes queries to appropriate handlers using semantic similarity.

Features:
1. Semantic Routing - Route based on query intent
2. Complexity-Based Routing - Simple vs complex query handling
3. Tool Selection - Choose appropriate retrieval method
"""

from semantic_router import Route
from semantic_router.routers import SemanticRouter
from retrieval_playground.utils import constants
from retrieval_playground.utils.model_manager import model_manager

# ============================================================================
# CONFIGURATION
# ============================================================================

ROUTE_SIMILARITY_THRESHOLD = 0.65
ROUTE_CONFIDENCE_PRECISION = 3


# ============================================================================
# ROUTE DEFINITIONS
# ============================================================================

greetings_route = Route(
    name="greetings",
    utterances=[
        # Greetings
        "hi", "hello", "hey there", "hi there", "good morning", "good afternoon", "good evening",
        # Status inquiries (about the assistant, not technical topics)
        "how are you", "how are you doing", "how's it going", "what's up", "how do you do",
        "what's new with you", "what's going on with you", "what's happening",
        # Gratitude
        "thank you", "thanks", "thank you very much", "appreciate it", "thanks a lot",
        # Farewells
        "goodbye", "bye", "see you", "see you later", "see you soon", "farewell", "take care",
        # Pleasantries
        "nice to meet you", "pleasure to meet you", "good to see you", "nice talking to you",
        # Capability inquiries (meta questions about the assistant)
        "what can you do", "what are your capabilities", "can you help me", "are you able to"
    ],
)

factual_route = Route(
    name="factual",
    utterances=[
        # Basic factual questions (what, who, when, where)
        "what is", "what are", "what was", "what were",
        "who is", "who are", "who was", "who created",
        "when was", "when did", "when were",
        "where is", "where are", "where was",
        # Quantitative factual questions
        "how many", "how much", "what year", "what date", "how long",
        # Listing and identification
        "list the", "list all", "name the", "identify the", "which is", "which are",
        # Information requests
        "tell me about", "give me information", "show me information about",
        # Definition queries
        "define", "definition of", "what does mean", "meaning of",
        "what is the definition", "terminology of", "what is a", "what are the",
        "explain the term", "what is meant by", "glossary term"
    ],
)

analytical_route = Route(
    name="analytical",
    utterances=[
        # Reasoning and causation
        "why does", "why do", "why is", "why are", "why would",
        "what causes", "what caused", "what leads to", "what led to",
        "reasoning behind", "reason for", "explain why",
        # Analysis and evaluation
        "analyze", "analyze the", "evaluate", "evaluate the", "assess", "assess the",
        "examine", "investigate", "discuss", "discuss the",
        # Impact and implications
        "impact of", "effect of", "implications of", "consequences of",
        "influence of", "result of", "outcome of",
        # Explanatory requests (not definitions)
        "explain how", "describe how", "describe the process",
        "elaborate on", "break down", "walk through the logic",
        # How does/do X work (complex explanatory)
        "how does", "how do", "how did", "how does it work", "how do they work",
        # Procedural how-to queries
        "how to", "how do I", "how can I", "how should I",
        "steps to", "process for", "procedure for", "method for",
        "guide to", "tutorial on", "instructions for", "way to",
        "approach to", "technique for", "strategy for"
    ],
)

comparison_route = Route(
    name="comparison",
    utterances=[
        # Direct comparison
        "compare", "compare the", "comparison of", "comparison between",
        "versus", "vs", "X versus Y", "X vs Y",
        # Difference queries
        "difference between", "differences between", "what's the difference",
        "how does X differ from Y", "how do X and Y differ",
        # Preference and quality comparison
        "which is better", "which is best", "better than", "worse than",
        "superior to", "inferior to", "preferable to",
        # Advantages and disadvantages
        "advantages and disadvantages", "pros and cons", "advantages vs disadvantages",
        "benefits and drawbacks", "strengths and weaknesses",
        # Performance and benchmarking
        "performance comparison", "benchmark comparison", "benchmark between",
        "faster than", "slower than", "more efficient",
        # Similarity comparison
        "similarities and differences", "similarities between", "how are X and Y similar",
        # Contrast
        "contrast", "contrast between", "contrasting X and Y"
    ],
)

# All routes
routes = [
    greetings_route,
    factual_route,
    analytical_route,
    comparison_route,
]


# ============================================================================
# SEMANTIC ROUTER INITIALIZATION
# ============================================================================

# Create encoder from model_manager (uses same embedding model)
encoder = model_manager.create_routing_encoder(score_threshold=ROUTE_SIMILARITY_THRESHOLD)

# Initialize semantic router
rl = SemanticRouter(
    encoder=encoder,
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
        "use_reranking": False,
        "description": "Casual conversation and greetings"
    },
    "factual": {
        "requires_retrieval": True,
        "retrieval_method": "hybrid_search",
        "tool": "vector_db",
        "use_reranking": False,
        "description": "Factual questions and definitions"
    },
    "analytical": {
        "requires_retrieval": True,
        "retrieval_method": "dense_search",
        "tool": "vector_db",
        "use_reranking": True,
        "description": "Analytical, reasoning, and how-to questions"
    },
    "comparison": {
        "requires_retrieval": True,
        "retrieval_method": "multi_query",
        "tool": "vector_db",
        "use_reranking": True,
        "description": "Comparison between entities or concepts"
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
        "use_reranking": False,
        "description": "Unknown route"
    })

    decision = {
        "route_name": route_name,
        "confidence": similarity_score,
        "requires_retrieval": metadata["requires_retrieval"],
        "retrieval_method": metadata["retrieval_method"],
        "tool": metadata["tool"],
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

    Routes determine INTENT (what kind: factual, analytical, comparison)
    Complexity determines DIFFICULTY (how hard: simple, moderate, complex)

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
    from retrieval_playground.utils.query_classifier import classify_query_complexity

    # Step 1: Semantic routing (what KIND of query?)
    route_decision = semantic_layer(query, return_metadata=True)

    # Step 2: Complexity analysis (how HARD is the query?)
    complexity_analysis = classify_query_complexity(query)

    # Step 3: Decide final method based on complexity
    # Simple queries: use route's suggested method
    # Moderate/Complex queries: use complexity's recommended strategy
    if complexity_analysis["complexity"] == "simple":
        final_retrieval_method = route_decision["retrieval_method"] or "dense_search"
    else:
        # Moderate or complex: trust complexity analysis
        strategy = complexity_analysis["recommended_strategy"]
        if strategy == "multi_query":
            final_retrieval_method = "multi_query"
        elif strategy == "step_back":
            final_retrieval_method = "hybrid_search"
        elif strategy == "decompose":
            final_retrieval_method = "multi_query"
        else:
            final_retrieval_method = "dense_search"

    return {
        "route": route_decision,
        "complexity": complexity_analysis,
        "final_strategy": complexity_analysis["recommended_strategy"],
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
        "embedding_model": f"{constants.EMBEDDING_MODEL_NAME} (via model_manager)",
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
    Demonstrate routing functionality with workshop dataset examples.
    """
    print("=" * 80)
    print("ROUTING DEMO - SciPy Workshop Dataset")
    print("=" * 80)

    # Test queries for each route type
    test_queries = [
        ("Hello! How are you?", "greetings"),
        ("What are the key components of the Agent Laboratory system?", "factual"),
        ("What is the definition of function space diffusion?", "factual"),
        ("Explain why quantum graph neural networks work better than classical methods", "analytical"),
        ("How can I train a transformer model on climate data?", "analytical"),
        ("Compare PyTorch versus JAX for scientific computing", "comparison"),
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
        print(f"Use Reranking: {result['use_reranking']}")
        print("-" * 80)

    # Test complexity-based routing
    print("\n" + "=" * 80)
    print("COMPLEXITY-BASED ROUTING")
    print("=" * 80)

    complex_queries = [
        "What is AutoClimDS?",
        "Compare the performance of different dataframe libraries and explain which is best for large-scale scientific data",
        "How do quantum graph neural networks leverage quantum mechanics to improve molecular property prediction compared to classical approaches?"
    ]

    for query in complex_queries:
        print(f"\nQuery: {query}")
        result = route_with_complexity_analysis(query)
        print(f"Route: {result['route']['route_name']}")
        print(f"Complexity: {result['complexity']['complexity']} (score: {result['complexity']['score']})")
        print(f"Final Strategy: {result['final_strategy']}")
        print(f"Final Retrieval Method: {result['final_retrieval_method']}")
        print(f"Signals: {result['metadata']['signals']}")

    # Test tool selection
    print("\n" + "=" * 80)
    print("TOOL SELECTION")
    print("=" * 80)

    tool_queries = [
        "What is function space diffusion?",
        "How many papers discuss climate data analysis?",
        "What are the latest developments in AI agents for scientific research in 2026?",
        "Compare the benchmark performance across different GPU architectures"
    ]

    for query in tool_queries:
        route_decision = route_with_complexity_analysis(query)
        tool = select_retrieval_tool(query, route_decision)
        print(f"\nQuery: {query}")
        print(f"Selected Tool: {tool}")
        print(f"Route: {route_decision['route']['route_name']}")

    # Test scientific paper queries
    print("\n" + "=" * 80)
    print("SCIENTIFIC PAPER QUERIES")
    print("=" * 80)

    scientific_queries = [
        "What experimental results demonstrate the effectiveness of AI agents in scientific research?",
        "Describe the architecture for climate data retrieval and analysis systems",
        "What optimization techniques are recommended for improving LLM performance in Python?"
    ]

    for query in scientific_queries:
        print(f"\nQuery: {query}")
        result = route_with_complexity_analysis(query)
        print(f"Route: {result['route']['route_name']}")
        print(f"Complexity: {result['complexity']['complexity']}")
        print(f"Strategy: {result['final_strategy']}")
        print(f"Reranking: {result['route']['use_reranking']}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demo_routing()
