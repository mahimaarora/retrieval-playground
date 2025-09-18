"""
Baseline RAG pipeline for document retrieval and question answering.
"""
from typing import List, Dict, Any

from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

from retrieval_playground.utils.model_manager import model_manager
from retrieval_playground.utils.pylogger import get_python_logger
from retrieval_playground.utils import constants
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import psutil
import os
import signal
from qdrant_client.models import Distance
from retrieval_playground.utils import config


class RAG:
    """RAG pipeline for document processing and question answering."""
    
    def __init__(self, strategy: str = None, use_cloud: bool = True):
        """
        Initialize the BaselineRAG pipeline.
        """
        self.logger = get_python_logger(log_level=constants.PYTHON_LOG_LEVEL)

        # Initialize LLM
        self.llm = model_manager.get_llm()
        self.embeddings = model_manager.get_embeddings()

        self.strategy = strategy
        if use_cloud:
            self.qdrant_path = None
            self.qdrant_client = QdrantClient(url=constants.QDRANT_URL, api_key=constants.QDRANT_KEY)
        else:   
            self.qdrant_path = config.QDRANT_DIR / self.strategy.value
            self.qdrant_client = QdrantClient(path=str(self.qdrant_path))

        # Define RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant that answers questions based on the provided context.
Please provide a comprehensive answer based on the context below. If the context doesn't contain enough information to answer the question, please say so.

Question: {question} 

Context:
{context}

Answer:"""
        )
        
        self.logger.info("BaselineRAG pipeline initialized")
    
    def retrieve_context(self, query: str, k: int = 5, collection_name: str = None) -> List[Document]:
        """
        Retrieve relevant context documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant Document objects
        """
        if collection_name is None:
            collection_name = self.strategy.value
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=self.embeddings,
            distance=Distance.COSINE 
        )
        results = vector_store.similarity_search_with_relevance_scores(query, k=3)
        return results
    
    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        Generate an answer using the LLM based on retrieved context.
        
        Args:
            query: User question
            context_docs: Retrieved context documents
            
        Returns:
            Generated answer string
        """
        # Format context from documents
        context = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in context_docs
        ])
        
        # Create the prompt
        messages = self.rag_prompt.format_messages(
            context=context,
            question=query
        )
        
        # Generate response
        self.logger.info(f"Generating answer for query: '{query[:50]}...'")
        response = self.llm.invoke(messages)
        
        return response.content
    
    def query(self, question: str, k: int = 5, collection_name: str = None) -> Dict[str, Any]:
        """
        Perform end-to-end RAG query: retrieve context and generate answer.
        
        Args:
            question: User question
            k: Number of context documents to retrieve
            
        Returns:
            Dictionary containing answer, context, and metadata
        """
        if collection_name is None:
            collection_name = self.strategy.value
        self.logger.info(f"Processing RAG query: '{question[:50]}...'")
        
        # Retrieve relevant context
        context_docs_with_score = self.retrieve_context(question, k=k, collection_name=collection_name)
        context_docs = [doc[0] for doc in context_docs_with_score]
        scores = [round(doc[1], 2) for doc in context_docs_with_score]
        
        # Generate answer
        answer = self.generate_answer(question, context_docs)
        
        # Prepare response
        response = {
            "question": question,
            "answer": answer,
            "scores": scores,
            "context": [
                {   
                    "scores": scores,
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", "Unknown")
                }
                for doc in context_docs
            ],
            "num_context_docs": len(context_docs)
        }
        
        self.logger.info("RAG query completed successfully")
        return response

    def close_qdrant_client(self):
        for p in psutil.process_iter(["pid", "cmdline"]):
            if p.info["cmdline"] and self.qdrant_path in " ".join(p.info["cmdline"]):
                os.kill(p.info["pid"], signal.SIGKILL)

        lock_file = os.path.join(self.qdrant_path, "lock")
        if os.path.exists(lock_file):
            os.remove(lock_file)
            