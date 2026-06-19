from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from typing import Dict, List, Any, Literal, Optional
import json

from retrieval_playground.utils import config, constants
from retrieval_playground.utils.model_manager import model_manager
from retrieval_playground.src.pre_retrieval.chunking_manager import ChunkingStrategy

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Optional imports for additional rerankers
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder as STCrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Type hints
RerankModel = Literal["huggingface", "bge", "flashrank"]


class Reranker:
    """
    Reranking retriever with multiple model options.

    Models:
    - "huggingface": Default cross-encoder (current)
    - "bge": BGE-reranker-v2-m3 (best free option, ~600MB)
    - "flashrank": Fast CPU reranking (<20ms)

    Example:
        # Default model
        reranker = Reranker()

        # Use BGE (best quality)
        reranker = Reranker(model="bge")

        # Use FlashRank (fastest)
        reranker = Reranker(model="flashrank")
    """

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.UNSTRUCTURED,
        use_cloud: bool = True,
        qdrant_client: QdrantClient = None,
        top_k: int = 20,
        top_n: int = 3,
        model: RerankModel = "huggingface"
    ):
        """
        Initialize reranker.

        Args:
            strategy: Chunking strategy
            use_cloud: Use cloud Qdrant (True) or local (False)
            qdrant_client: Optional pre-configured Qdrant client
            top_k: Number of documents to retrieve before reranking
            top_n: Number of documents to return after reranking
            model: Which reranker model to use
        """
        self.strategy = strategy
        self.top_k = top_k
        self.top_n = top_n
        self.model = model

        if use_cloud:
            self.qdrant_client = QdrantClient(url=constants.QDRANT_URL, api_key=constants.QDRANT_KEY)
        elif qdrant_client is None:
            self.qdrant_path = config.QDRANT_DIR / self.strategy.value
            self.qdrant_client = QdrantClient(path=str(self.qdrant_path))
        else:
            self.qdrant_client = qdrant_client

        self._setup_reranker_retriever()
        print(f"✅ Reranker initialized (model: {model})")
    
    def _setup_reranker_retriever(self):
        """Initialize the reranking retriever."""

        self.embeddings = model_manager.get_embeddings()
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.strategy.value,
            embedding=self.embeddings,
        )

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})

        # Initialize model based on selection
        if self.model == "huggingface":
            # Default HuggingFace cross-encoder (current implementation)
            model = HuggingFaceCrossEncoder(
                model_name=constants.RERANKER_MODEL,
                model_kwargs={"trust_remote_code": True}
            )
            compressor = CrossEncoderReranker(model=model, top_n=self.top_n)
            self.reranker_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=self.retriever
            )

        elif self.model == "bge":
            # BGE Reranker v2-m3 (best free option)
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers required for BGE reranker. "
                    "Install with: pip install sentence-transformers"
                )
            self.bge_model = STCrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
            self.reranker_retriever = None  # Use custom rerank method

        elif self.model == "flashrank":
            # FlashRank (fast CPU)
            if not FLASHRANK_AVAILABLE:
                raise ImportError(
                    "flashrank required for FlashRank reranker. "
                    "Install with: pip install flashrank"
                )
            self.flashrank_model = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
            self.reranker_retriever = None  # Use custom rerank method

        else:
            raise ValueError(f"Unknown model: {self.model}")
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve and rerank documents for a query.

        Args:
            query: Search query

        Returns:
            Top-n reranked documents
        """
        # Get initial documents
        initial_docs = self.retriever.invoke(query)

        if self.model == "huggingface":
            # Use LangChain's reranker retriever
            return self.reranker_retriever.invoke(query)

        elif self.model == "bge":
            # BGE reranking
            return self._rerank_bge(initial_docs, query)

        elif self.model == "flashrank":
            # FlashRank reranking
            return self._rerank_flashrank(initial_docs, query)

    def _rerank_bge(self, documents: List[Document], query: str) -> List[Document]:
        """Rerank using BGE model."""
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Get scores
        scores = self.bge_model.predict(pairs)

        # Sort by score
        doc_scores = list(zip(documents, scores))
        sorted_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:self.top_n]

        # Add metadata
        reranked_docs = []
        for doc, score in sorted_docs:
            doc.metadata["rerank_score"] = float(score)
            doc.metadata["reranker"] = "bge"
            reranked_docs.append(doc)

        return reranked_docs

    def _rerank_flashrank(self, documents: List[Document], query: str) -> List[Document]:
        """Rerank using FlashRank."""
        if not documents:
            return []

        # Prepare passages
        passages = [
            {"id": i, "text": doc.page_content}
            for i, doc in enumerate(documents)
        ]

        # Rerank
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.flashrank_model.rerank(rerank_request)

        # Sort and take top_n
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)[:self.top_n]

        # Map back to documents
        reranked_docs = []
        for result in sorted_results:
            doc = documents[result["id"]]
            doc.metadata["rerank_score"] = float(result["score"])
            doc.metadata["reranker"] = "flashrank"
            reranked_docs.append(doc)

        return reranked_docs
    
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
    