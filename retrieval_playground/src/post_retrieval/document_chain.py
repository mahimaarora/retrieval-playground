# LangChain imports
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import LLMChain, RefineDocumentsChain, MapRerankDocumentsChain, MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.output_parsers.regex import RegexParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangGraph imports
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

# Python imports
import operator
from typing import Dict, List, Any, Literal, TypedDict, Annotated

# Project imports
from retrieval_playground.utils.model_manager import model_manager
from retrieval_playground.utils import config

# Initialize llm instance
llm = model_manager.get_llm()


def setup_stuff_chain():
    """Set up a simple stuff documents chain that concatenates all documents."""
    stuff_prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant that answers questions based on the provided context.
        Please provide a comprehensive answer based on the context below. 
        If the context doesn't contain enough information to answer the question, please say so.
    
        Question: {question} 
    
        Context:
        {context}
    
        Answer:"""
    )
    
    stuff_chain = create_stuff_documents_chain(llm, stuff_prompt)
    return stuff_chain


def setup_refine_chain():
    """Set up a refine documents chain that iteratively improves the answer."""
    document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )
    
    # Initial summarization prompt
    summarize_prompt = ChatPromptTemplate.from_messages([
        ("human", 
        """You are a helpful assistant that answers questions based on the given context.
    
        Question: {question}
    
        Context:
        {context}
    
        Provide the best possible answer based on this context:""")
    ])
    initial_llm_chain = LLMChain(llm=llm, prompt=summarize_prompt)
    
    # Refinement prompt
    refine_template = """
    We have an existing answer so far:
    {existing_answer}
    
    Here is some new context:
    ------------
    {context}
    ------------
    
    Refine the existing answer where needed, keeping it accurate and comprehensive.
    If the new context is not useful, keep the answer unchanged.
    """
    refine_prompt = ChatPromptTemplate.from_messages([("human", refine_template)])
    refine_llm_chain = LLMChain(llm=llm, prompt=refine_prompt)
    
    # Build the refine chain
    refine_chain = RefineDocumentsChain(
        initial_llm_chain=initial_llm_chain,
        refine_llm_chain=refine_llm_chain,
        document_prompt=document_prompt,
        document_variable_name="context",
        initial_response_name="existing_answer",
    )
    return refine_chain

def setup_map_rerank_chain():
    """Set up a map-rerank chain that scores and ranks document responses."""
    prompt_template = """
    You are a helpful assistant. 
    Answer the following question using ONLY the given context. 
    If the context does not contain the answer, say "Not enough information."

    Question: {question}

    Context:
    {context}

    Provide your answer and a confidence score (1-10) in this format:
    <Answer>
    Score: <Score>
    """

    # Parser for extracting answer and score
    output_parser = RegexParser(
        regex=r"(?s)(.*?)\n+Score:\s*([0-9]+)",
        output_keys=["answer", "score"],
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
        output_parser=output_parser,
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Rerank chain
    rerank_chain = MapRerankDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        rank_key="score",
        answer_key="answer",
    )
    return rerank_chain

def setup_map_reduce_chain():
    """Set up a map-reduce chain that summarizes documents then combines summaries."""
    # Map prompt - summarizes each document
    map_template = "Write a concise summary of the following: {docs}"
    map_prompt = ChatPromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce prompt - combines summaries and answers question
    reduce_template = """The following is a set of summaries:
    {docs}

    Based on these summaries, answer the question: {question}"""
    reduce_prompt = ChatPromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Combine documents chain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, 
        document_variable_name="docs"
    )

    # Reduce documents chain
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=1000,
    )

    # Create the map-reduce chain
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )

    return map_reduce_chain


def setup_refine_chain_langgraph():
    """Set up a LangGraph-based refine chain that iteratively improves answers."""
    
    # Initial answer generation prompt
    initial_prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant that answers questions based on the given context.
    
    Question: {question}
    
    Context:
    {context}
    
    Provide the best possible answer based on this context."""
    )
    initial_chain = initial_prompt | llm | StrOutputParser()
    
    # Answer refinement prompt
    refine_prompt = ChatPromptTemplate.from_template(
        """We have an existing answer so far:
    {existing_answer}
    
    Here is some new context:
    ------------
    {context}
    ------------
    
    Refine the existing answer where needed, keeping it accurate and comprehensive.
    If the new context is not useful, keep the answer unchanged."""
    )
    refine_chain = refine_prompt | llm | StrOutputParser()
    
    class RefineState(TypedDict):
        """State for the refine chain workflow."""
        question: str
        docs: List[str]
        index: int
        answer: str
    
    def generate_initial_answer(state: RefineState) -> Dict[str, Any]:
        """Generate the first answer from the first document."""
        first_doc = state["docs"][0]
        answer = initial_chain.invoke({
            "question": state["question"], 
            "context": first_doc
        })
        return {"answer": answer, "index": 1}
    
    def refine_answer(state: RefineState) -> Dict[str, Any]:
        """Refine the current answer with the next document."""
        doc = state["docs"][state["index"]]
        refined_answer = refine_chain.invoke({
            "existing_answer": state["answer"], 
            "context": doc
        })
        return {"answer": refined_answer, "index": state["index"] + 1}
    
    def should_refine(state: RefineState) -> Literal["refine_answer", END]:
        """Determine whether to continue refining or end the process."""
        if state["index"] >= len(state["docs"]):
            return END
        return "refine_answer"
    
    # Create and configure the graph
    refine_graph = StateGraph(RefineState)
    refine_graph.add_node("generate_initial_answer", generate_initial_answer)
    refine_graph.add_node("refine_answer", refine_answer)
    
    refine_graph.add_edge(START, "generate_initial_answer")
    refine_graph.add_conditional_edges("generate_initial_answer", should_refine)
    refine_graph.add_conditional_edges("refine_answer", should_refine)
    
    refine_app = refine_graph.compile()
    return refine_app

def setup_map_rerank_chain_langgraph():
    """Set up a LangGraph-based map-rerank chain that scores and ranks document responses."""
    
    class AnswerWithScore(TypedDict):
        answer: str
        score: Annotated[int, ..., "Score from 1-10."]
    
    prompt_template = """
    You are a helpful assistant. 
    Answer the following question using ONLY the given context. 
    If the context does not contain the answer, say "Not enough information."
    
    Question: {question}
    
    Context:
    {context}
    
    Provide your answer and a confidence score (1-10) in this format:
    <Answer>
    Score: <Score>
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    map_chain = prompt | llm.with_structured_output(AnswerWithScore)
    
    class State(TypedDict):
        contents: List[str]
        question: str                      
        answers_with_scores: Annotated[list, operator.add]
        answer: str
        
    class MapState(TypedDict):
        content: str
        question: str
    
    def map_analyses(state: State):
        """Generate a node for each document."""
        return [
            Send("generate_analysis", {"content": content, "question": state["question"]})
            for content in state["contents"]
        ]
    
    def generate_analysis(state: MapState):
        """Generate answer and score for a document."""
        response = map_chain.invoke({
            "context": state["content"], 
            "question": state["question"]
        })
        return {"answers_with_scores": [response]}
    
    def pick_top_ranked(state: State):
        """Select the highest-scoring answer."""
        ranked_answers = sorted(
            state["answers_with_scores"], key=lambda x: -int(x["score"])
        )
        return {"answer": ranked_answers[0]}
    
    # Create and configure the graph
    rerank_graph = StateGraph(State)
    rerank_graph.add_node("generate_analysis", generate_analysis)
    rerank_graph.add_node("pick_top_ranked", pick_top_ranked)
    rerank_graph.add_conditional_edges(START, map_analyses, ["generate_analysis"])
    rerank_graph.add_edge("generate_analysis", "pick_top_ranked")
    rerank_graph.add_edge("pick_top_ranked", END)
    
    rerank_app = rerank_graph.compile()
    return rerank_app

def setup_map_reduce_chain_langgraph():
    """Set up a LangGraph-based map-reduce chain that summarizes documents then combines summaries."""
    
    map_template = "Write a concise summary of the following: {context}"
    reduce_template = """The following is a set of summaries:
    {docs}
    
    Based on these summaries, answer the question: {question}"""
    
    map_prompt = ChatPromptTemplate.from_template(map_template)
    reduce_prompt = ChatPromptTemplate.from_template(reduce_template)
    
    map_chain = map_prompt | llm | StrOutputParser()
    reduce_chain = reduce_prompt | llm | StrOutputParser()
    
    class OverallState(TypedDict):
        contents: List[str]
        summaries: Annotated[list, operator.add]
        question: str
        final_answer: str
    
    class SummaryState(TypedDict):
        content: str
    
    def generate_summary(state: SummaryState):
        """Generate summary for a document."""
        response = map_chain.invoke({"context": state["content"]})
        return {"summaries": [response]}
    
    def map_summaries(state: OverallState):
        """Map documents to summary nodes."""
        return [
            Send("generate_summary", {"content": content}) for content in state["contents"]
        ]
    
    def generate_final_answer(state: OverallState):
        """Generate final answer from summaries."""
        response = reduce_chain.invoke({
            "docs": "\n".join(state["summaries"]), 
            "question": state["question"]
        })
        return {"final_answer": response}
    
    # Create and configure the graph
    map_reduce_graph = StateGraph(OverallState)
    map_reduce_graph.add_node("generate_summary", generate_summary)
    map_reduce_graph.add_node("generate_final_answer", generate_final_answer)
    map_reduce_graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    map_reduce_graph.add_edge("generate_summary", "generate_final_answer")
    map_reduce_graph.add_edge("generate_final_answer", END)
    
    map_reduce_app = map_reduce_graph.compile()
    return map_reduce_app

