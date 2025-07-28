from typing import Dict, Literal
from langgraph.graph import StateGraph, END, START
from src.agentic_nodes import (
    ingest_question,
    check_relevance,
    check_memory,
    rewrite_question,
    create_chunks,
    generate_answer_per_chunks,
    generate_synthetized_answer,
    update_memory,
    output_answer,
)
from src.agentic_state import AgenticState


def build_agentic_graph() -> StateGraph:
    """
    Construct and return the compiled agentic LangGraph for feedback analysis.
    """
    agentic_graph = StateGraph(AgenticState)
    
    # Nodes
    agentic_graph.add_node("ingest_question", ingest_question)
    agentic_graph.add_node("check_relevance", check_relevance)
    agentic_graph.add_node("check_memory", check_memory)
    agentic_graph.add_node("rewrite_question", rewrite_question)
    agentic_graph.add_node("create_chunks", create_chunks)
    agentic_graph.add_node("generate_answer_per_chunks", generate_answer_per_chunks)
    agentic_graph.add_node("generate_synthetized_answer", generate_synthetized_answer)
    agentic_graph.add_node("update_memory", update_memory)
    agentic_graph.add_node("output_answer", output_answer)
    
    # Edges
    agentic_graph.add_edge(START, "ingest_question") 
    agentic_graph.add_edge("ingest_question", "check_relevance")
    
    def route_relevance(state: AgenticState) -> Literal["irrelevant", "relevant"]:
        return "relevant" if state["is_relevant"] else "irrelevant"
    
    agentic_graph.add_conditional_edges(
        "check_relevance",
        route_relevance,
        {
            "irrelevant": "output_answer",      
            "relevant": "check_memory",
        },
    )
    
    def route_memory(state: AgenticState) -> Literal["cached", "not_cached"]:
        return "cached" if state.get("from_memory") else "not_cached"
    
    agentic_graph.add_conditional_edges(
        "check_memory",
        route_memory,
        {
            "cached": "output_answer",
            "not_cached": "rewrite_question",
        },
    )
    
    agentic_graph.add_edge("rewrite_question", "create_chunks")
    agentic_graph.add_edge("create_chunks", "generate_answer_per_chunks")
    agentic_graph.add_edge("generate_answer_per_chunks", "generate_synthetized_answer")
    agentic_graph.add_edge("generate_synthetized_answer", "update_memory")
    agentic_graph.add_edge("update_memory", "output_answer")
    agentic_graph.add_edge("output_answer", END)
    
    return agentic_graph
