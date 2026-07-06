"""
Query Classifier

Analyzes query complexity using rule-based signals.
No LLM needed - fast, deterministic classification.
"""


def classify_query_complexity(query: str) -> dict:
    """
    Classify query complexity using 5 signals.

    Signals:
    1. Length (> 15 words) → +1
    2. Multi-hop indicators → +2 (highest weight)
    3. Multiple questions → +1
    4. Comparison request → +1
    5. Analytical/reasoning → +1

    Args:
        query: Query string to classify

    Returns:
        {
            "complexity": "simple" | "moderate" | "complex",
            "score": int (0-5+),
            "signals": {
                "length": bool,
                "multi_hop": bool,
                "multiple_questions": bool,
                "comparison": bool,
                "analytical": bool
            },
            "recommended_strategy": "standard_rag" | "multi_query" | "step_back" | "decompose"
        }

    Example:
        >>> classify_query_complexity("What is RAG?")
        {"complexity": "simple", "score": 0, ...}

        >>> classify_query_complexity("Compare BERT and GPT architectures")
        {"complexity": "simple", "score": 1, ...}
    """
    complexity_score = 0
    signals = {
        "length": False,
        "multi_hop": False,
        "multiple_questions": False,
        "comparison": False,
        "analytical": False
    }

    # Signal 1: Query length (> 15 words)
    word_count = len(query.split())
    if word_count > 15:
        complexity_score += 1
        signals["length"] = True

    # Signal 2: Multi-hop indicators (highest weight: +2)
    # These require connecting multiple pieces of information
    # NOTE: Avoid overlap with analytical (Signal 5) - don't include starting phrases
    multi_hop_markers = [
        # Comparative context
        "compared to", "in the context of", "in relation to", "relative to",
        # Relationships and connections
        "relationship between", "connection between", "correlation between",
        "interaction between", "interplay between", "link between",
        # Working mechanisms (complex explanations)
        "work in", "works in", "working in", "function in", "functions in",
        "mechanism", "mechanisms", "process of", "processes of",
        # Effects and impacts (cause-chain)
        "what effect", "what impact", "what influence", "what role",
        "impact on", "influence on", "role in", "contribute to",
        # Reasoning depth (mid-query patterns)
        "reasoning behind", "rationale for", "basis for"
    ]
    if any(marker in query.lower() for marker in multi_hop_markers):
        complexity_score += 2
        signals["multi_hop"] = True

    # Signal 3: Multiple questions
    if query.count("?") > 1 or (query.count(" and ") > 1 and query.count("?") >= 1):
        complexity_score += 1
        signals["multiple_questions"] = True

    # Signal 4: Comparison request (weight: +2, same as multi-hop)
    # Comparisons are inherently multi-faceted (need to cover both entities)
    comparison_markers = [
        # Direct comparison
        "compare", "comparison", "comparison of", "comparison between",
        "versus", "vs", "vs.", "vs ", " v ", " v. ",
        # Differences and similarities
        "difference between", "differences between", "differ from",
        "distinguish between", "differentiate between",
        "similar to", "similarity", "similarities between",
        "contrast", "contrasting", "contrast between",
        # Preference and evaluation
        "better than", "worse than", "superior to", "inferior to",
        "which is better", "which is best", "which is worse",
        "more effective", "less effective", "more efficient", "less efficient",
        "faster than", "slower than", "cheaper than", "more expensive",
        # Advantages/disadvantages
        "pros and cons", "advantages and disadvantages",
        "strengths and weaknesses", "benefits and drawbacks",
        "tradeoffs", "trade-offs", "trade offs"
    ]
    if any(marker in query.lower() for marker in comparison_markers):
        complexity_score += 2
        signals["comparison"] = True

    # Signal 5: Analytical/reasoning request (must START with these words)
    # NOTE: These use .startswith() - only trigger if query BEGINS with these
    analytical_markers = [
        # Why questions (causation) - removed duplicates like "why does", "why do" (covered by "why")
        "why",
        # How questions (process/mechanism) - removed duplicates like "how does", "how do" (covered by "how")
        "how",
        # Explanation requests
        "explain", "describe", "elaborate", "clarify",
        # Analysis requests
        "analyze", "analyse", "evaluate", "assess", "examine", "investigate",
        # Discussion and critique
        "discuss", "critique", "review",
        # Reasoning
        "justify", "argue", "demonstrate", "prove", "show that"
    ]
    if any(query.lower().startswith(marker) for marker in analytical_markers):
        complexity_score += 1
        signals["analytical"] = True

    # Determine complexity level and recommended strategy
    if complexity_score <= 1:
        complexity = "simple"
        recommended_strategy = "standard_rag"
    elif complexity_score == 2:
        complexity = "moderate"
        if signals["comparison"]:
            recommended_strategy = "multi_query"
        else:
            recommended_strategy = "step_back"
    else:
        complexity = "complex"
        if signals["multiple_questions"]:
            recommended_strategy = "decompose"
        elif signals["multi_hop"]:
            recommended_strategy = "step_back"
        else:
            recommended_strategy = "multi_query"

    return {
        "complexity": complexity,
        "score": complexity_score,
        "signals": signals,
        "recommended_strategy": recommended_strategy
    }
