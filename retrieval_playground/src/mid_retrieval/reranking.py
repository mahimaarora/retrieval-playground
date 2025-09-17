from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from typing import Dict, List, Any
import json

from retrieval_playground.utils import config, constants
from retrieval_playground.utils.model_manager import model_manager
from retrieval_playground.src.pre_retrieval.chunking_strategies import ChunkingStrategy

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Reranker:
    """Reranking retriever using cross-encoder models."""
    
    def __init__(self, strategy: ChunkingStrategy = ChunkingStrategy.UNSTRUCTURED, use_cloud: bool = True, qdrant_client: QdrantClient = None, top_k: int = 20, top_n: int = 3):
        self.strategy = strategy
        self.top_k = top_k
        self.top_n = top_n
        if use_cloud:
            self.qdrant_client = QdrantClient(url=constants.QDRANT_URL, api_key=constants.QDRANT_KEY)
        elif qdrant_client is None:
            self.qdrant_path = config.QDRANT_DIR / self.strategy.value
            self.qdrant_client = QdrantClient(path=str(self.qdrant_path))
        else:
            self.qdrant_client = qdrant_client
        self._setup_reranker_retriever()
        print("✅ Reranker initialized")
    
    def _setup_reranker_retriever(self):
        """Initialize the reranking retriever."""
        
        self.embeddings = model_manager.get_embeddings()
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.strategy.value,
            embedding=self.embeddings,
        )

        self.retriever = vector_store.as_retriever(search_kwargs={"k": self.top_k})

        model = HuggingFaceCrossEncoder(model_name=constants.RERANKER_MODEL)
        compressor = CrossEncoderReranker(model=model, top_n=self.top_n)
        self.reranker_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.retriever
        )
    
    def retrieve(self, query: str):
        """Retrieve and rerank documents for a query."""
        return self.reranker_retriever.invoke(query)
    
    @staticmethod
    def load_test_queries() -> List[Dict[str, Any]]:
        """Load test queries from JSON file."""
        queries_path = config.TESTS_DIR / "test_queries.json"
        with open(queries_path, 'r') as f:
            return json.load(f)
            
    def evaluate_reranking(self, close_qdrant_client: bool = True) -> Dict[str, float]:
        """Evaluate reranking performance against baseline retrieval."""
        print("Starting reranking evaluation...")
        test_queries = self.load_test_queries()
        print(f"Evaluating {len(test_queries)} test queries")
        
        reranker_scores = []
        retriever_scores = []
        
        for idx, query in enumerate(test_queries):
            print(f"\nQuery {idx+1}: {query['user_input'][:100]}...")
            
            reranker_output = self.reranker_retriever.invoke(query["user_input"])
            retriever_output = self.retriever.invoke(query["user_input"])[:self.top_n]
            
            query_reranker_scores = []
            query_retriever_scores = []
            
            for i in range(min(self.top_n, len(reranker_output), len(retriever_output))):
                reference_context = query["reference_context"]
                reranker_content = reranker_output[i].page_content
                retriever_content = retriever_output[i].page_content
                
                # Convert to embeddings
                ref_embedding = self.embeddings.embed_query(reference_context)
                reranker_embedding = self.embeddings.embed_query(reranker_content)
                retriever_embedding = self.embeddings.embed_query(retriever_content)
                
                # Calculate cosine similarities
                reranker_score = cosine_similarity([ref_embedding], [reranker_embedding])[0][0]
                retriever_score = cosine_similarity([ref_embedding], [retriever_embedding])[0][0]
                
                print(f"  Result {i+1}: Reranker={reranker_score:.3f}, Baseline={retriever_score:.3f}")
                
                query_reranker_scores.append(reranker_score)
                query_retriever_scores.append(retriever_score)
                reranker_scores.append(reranker_score)
                retriever_scores.append(retriever_score)
            
            query_avg_reranker = np.mean(query_reranker_scores)
            query_avg_retriever = np.mean(query_retriever_scores)
            print(f"  📊 Query avg: Reranker={query_avg_reranker:.3f}, Baseline={query_avg_retriever:.3f}")
        
        final_results = {
            "reranker_avg_score": np.round(np.mean(reranker_scores), 4),
            "retriever_avg_score": np.round(np.mean(retriever_scores), 4),
            "improvement": np.round(np.mean(reranker_scores) - np.mean(retriever_scores), 4)
        }
        if close_qdrant_client:
            self.close_qdrant_client()
        
        print("\n" + "="*50)
        print("🎯 FINAL RESULTS")
        print("="*50)
        print(f"Reranker Average Score:  {final_results['reranker_avg_score']:.4f}")
        print(f"Baseline Average Score:  {final_results['retriever_avg_score']:.4f}")
        print(f"Improvement:             {final_results['improvement']:.4f}")
        improvement_pct = (final_results['improvement'] / final_results['retriever_avg_score']) * 100
        print(f"Improvement Percentage:  {improvement_pct:.2f}%")
        
        if final_results['improvement'] > 0:
            print("✅ Reranking shows improvement!")
        else:
            print("❌ Reranking shows no improvement")
        
        return final_results

    def close_qdrant_client(self):
        """Close the Qdrant client."""
        self.qdrant_client.close()

if __name__ == "__main__":
    strategy = ChunkingStrategy.UNSTRUCTURED
    qdrant_path = config.QDRANT_DIR / strategy.value
    qdrant_client = QdrantClient(path=str(qdrant_path))
    reranker = Reranker(strategy=strategy, qdrant_client=qdrant_client, top_n=2)
    reranker.evaluate_reranking()
    