from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.routers import SemanticRouter
from retrieval_playground.utils import constants

# Routing configuration constants
ROUTE_SIMILARITY_THRESHOLD = 0.7
ROUTE_CONFIDENCE_PRECISION = 3

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

papers_route = Route(
    name="research_papers",
    utterances=[
        # Analytics & Causal Analysis (aligned with Analytics_2025_MC3G paper)
        "research papers in analytics", "causal analysis", "counterfactual generation",
        "model agnostic methods", "causally constrained models", "causal inference",
        "analytics research", "data analytics", "predictive analytics",
        
        # Computer Vision & Remote Sensing (aligned with Computer_Vision_2025 paper)
        "research papers in computer vision", "computer vision studies", "image segmentation",
        "remote sensing images", "annotation-free segmentation", "open-vocabulary segmentation",
        "image processing techniques", "computer vision algorithms", "visual recognition",
        "satellite imagery analysis", "geospatial analysis",
        
        # Generative AI & Computational Complexity (aligned with Generative_AI_2025 paper)
        "research papers in generative AI", "generative AI studies", "computational complexity",
        "state space models", "satisfiability problems", "generative modeling",
        "AI model complexity", "algorithmic complexity", "generative algorithms",
        
        # Machine Learning & Manifold Learning (aligned with Machine_Learning_2025 paper)
        "research papers in machine learning", "machine learning research", "manifold learning",
        "Riemannian geometry", "change point detection", "robust centroid estimation",
        "statistical learning", "unsupervised learning", "geometric machine learning",
        "manifold analysis", "topological data analysis",
        
        # Statistics & Publication Bias (aligned with Statistics_2025 paper)
        "research papers in statistics", "statistical research", "publication bias",
        "Copas-Jackson bounds", "meta-analysis", "selection bias", "statistical inference",
        "bias correction methods", "statistical methodology", "research methodology",
        
        # General academic and technical queries
        "what is machine learning", "explain deep learning", "neural networks",
        "artificial intelligence", "AI research", "data science",
        "academic papers", "scientific publications", "research findings",
        "literature review", "survey paper", "technical documentation",
        "methodology", "experimental results", "empirical studies",
        "peer-reviewed research", "academic research", "scientific studies"
    ],
)


routes = [greetings_route, papers_route]

rl = SemanticRouter(encoder=HuggingFaceEncoder(name=constants.DEFAULT_EMBEDDING_MODEL, score_threshold=ROUTE_SIMILARITY_THRESHOLD), routes=routes, auto_sync="local")


def get_route_info():
    """
    Get information about available routes and their configurations.
    
    Returns:
        Dict containing route information and statistics
    """
    route_info = {
        "total_routes": len(routes),
        "routes": {
            "greetings": {
                "name": "greetings",
                "utterances_count": len(greetings_route.utterances),
                "description": "Handles casual conversation and greetings"
            },
            "research_papers": {
                "name": "research_papers", 
                "utterances_count": len(papers_route.utterances),
                "description": "Routes academic and research-related queries"
            }
        },
        "embedding_model": constants.DEFAULT_EMBEDDING_MODEL,
        "default_threshold": ROUTE_SIMILARITY_THRESHOLD
    }
    return route_info

def greetings():
    return (
        "ROUTING: Query classified as greeting/casual conversation. "
        "No retrieval required - proceeding directly to LLM for response generation."
    )

def research_papers():
    return (
        "ROUTING: Query classified as research-related. "
        "Retrieval enabled - routing to Research Papers vector database for context extraction."
    )

def semantic_layer(query: str, similarity_threshold: float = ROUTE_SIMILARITY_THRESHOLD):
    """
    Apply semantic routing to determine query handling strategy.
    
    Args:
        query: Input query string
        similarity_threshold: Minimum similarity score for route activation
        
    Returns:
        Enhanced query string with system routing notes
    """
    try:
        route = rl(query)
        print(f"Route for the query {query}: {route}")
        if route.similarity_score is None:
            system_note = (
                f"ROUTING: Low confidence match. No route found for the query. "
                f"Provide default response for the query."
            )
            enhanced_query = f"{query} [SYSTEM_NOTE: {system_note}]"
            return enhanced_query

        similarity_score = round(route.similarity_score, ROUTE_CONFIDENCE_PRECISION)
            
        if route.name == "greetings":
            system_note = f"{greetings()} [Confidence: {similarity_score}]"
        elif route.name == "research_papers":
            system_note = f"{research_papers()} [Confidence: {similarity_score}]"
        else:
            # Fallback for unrecognized routes
            system_note = (
                f"ROUTING: Unrecognized route '{route.name}'. "
                f"Provide default response for the query. [Score: {similarity_score}]"
            )
        
        enhanced_query = f"{query} [SYSTEM_NOTE: {system_note}]"
        return enhanced_query
        
    except Exception as e:
        # Fallback in case of routing errors
        fallback_note = (
            f"ROUTING: Error in semantic routing ({str(e)}). "
            f"Provide default response for the query."
        )
        return f"{query} [SYSTEM_NOTE: {fallback_note}]"


def demonstrate_routing_examples():
    """
    Demonstrate routing functionality with 5 different example queries.
    
    Returns:
        List of dictionaries containing example queries and their routing results
    """
    examples = [
        # Example 1: Greeting/General query
        {
            "category": "greetings",
            "query": "Hi there! How are you doing today?",
            "description": "Casual greeting that should route to greetings handler"
        },
        
        # Example 2: Another greeting/general query  
        {
            "category": "greetings",
            "query": "Thank you for your help with this research!",
            "description": "Gratitude expression that should route to greetings handler"
        },
        
        # Example 3: Papers route - Analytics/Causal Analysis
        {
            "category": "research_papers",
            "query": "What research papers discuss counterfactual generation and causal analysis methods?",
            "description": "Analytics research query aligned with Analytics_2025_MC3G paper"
        },
        
        # Example 4: Papers route - Computer Vision/Remote Sensing
        {
            "category": "research_papers", 
            "query": "Can you explain annotation-free segmentation techniques for remote sensing images?",
            "description": "Computer vision query aligned with Computer_Vision_2025 paper"
        },
        
        # Example 5: Default/Low confidence query
        {
            "category": "default_fallback",
            "query": "Give cheesecake recipe",
            "description": "Unrelated query that should trigger default fallback routing"
        }
    ]
    
    results = []
    print("\n\n🚀 SEMANTIC ROUTING EXAMPLES DEMONSTRATION\n" + "="*60)
    
    for i, example in enumerate(examples, 1):
        print(f"\n📝 EXAMPLE {i}: {example['category'].upper()}")
        print(f"Query: \"{example['query']}\"")
        print(f"Expected: {example['description']}")
        
        # Process the query through semantic routing
        try:
            enhanced_query = semantic_layer(example['query'])
            route = rl(example['query'])
            if route.similarity_score is not None:
                similarity_score = round(route.similarity_score, ROUTE_CONFIDENCE_PRECISION)
            else:
                similarity_score = None

            result = {
                "example_number": i,
                "category": example['category'],
                "original_query": example['query'],
                "enhanced_query": enhanced_query,
                "detected_route": route.name if route else "None",
                "similarity_score": similarity_score,
                "description": example['description'],
                "routing_successful": True
            }
            
            print(f"Detected Route: {route.name}")
            print(f"Confidence Score: {similarity_score}")
            print(f"Query with Note: {enhanced_query}")
            
        except Exception as e:
            result = {
                "example_number": i,
                "category": example['category'],
                "original_query": example['query'],
                "enhanced_query": f"{example['query']} [SYSTEM_NOTE: Error in routing]",
                "detected_route": "error",
                "similarity_score": 0.0,
                "description": example['description'],
                "routing_successful": False,
                "error": str(e)
            }
            print(f"❌ Routing Error: {str(e)}")
        
        results.append(result)
        print("-" * 60)
    
    return results


def run_routing_examples():
    """
    Run and display routing examples for testing and demonstration.
    """
    print("🧪 Testing Semantic Routing System...")
    print(f"Route Configuration: {len(routes)} routes available")
    print(f"Similarity Threshold: {ROUTE_SIMILARITY_THRESHOLD}")
    print(f"Embedding Model: {constants.DEFAULT_EMBEDDING_MODEL}")
    
    # Run the examples
    results = demonstrate_routing_examples()
    
    # Summary statistics
    successful_routes = sum(1 for r in results if r['routing_successful'])
    print(f"\n📈 ROUTING SUMMARY:")
    print(f"Total Examples: {len(results)}")
    print(f"Successful Routes: {successful_routes}")
    print(f"Success Rate: {(successful_routes/len(results)*100):.1f}%")
    
    return results

if __name__ == "__main__":
    run_routing_examples()
    