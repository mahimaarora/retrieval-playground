"""
Document assembly: concatenate prepared chunks for generation.

Production RAG typically uses a single "stuff" prompt after context preparation.
"""

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from retrieval_playground.utils.model_manager import model_manager

llm = model_manager.get_llm()

STUFF_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers questions based on the provided context.
Please provide a comprehensive answer based on the context below.
If the context doesn't contain enough information to answer the question, please say so.

Question: {question}

Context:
{context}

Answer:"""
)


def setup_stuff_chain():
    """Create a stuff-documents chain for final answer generation."""
    return create_stuff_documents_chain(llm, STUFF_PROMPT)


def assemble_context(chunks: list[Document], separator: str = "\n\n") -> str:
    """Concatenate prepared chunks into a single context string."""
    return separator.join(chunk.page_content for chunk in chunks)


def generate_answer(question: str, chunks: list[Document]) -> str:
    """Assemble context and generate an answer in one step."""
    chain = setup_stuff_chain()
    return chain.invoke({"context": chunks, "question": question})
