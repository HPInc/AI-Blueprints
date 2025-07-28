from typing import TypedDict, List, Dict, Any, Optional
from langchain_community.llms import LlamaCpp
from langchain.docstore.document import Document
from src.simple_kv_memory import SimpleKVMemory


class AgenticState(TypedDict, total=False):
    """
    Represents the internal state of an agentic feedback analysis pipeline.
    This state is passed through the LangGraph nodes.
    """

    # Input metadata
    topic: str
    question: str

    # Document & chunking
    docs: str                      # Raw or combined document content
    chunks: str                    # Raw or processed chunks

    # LLM configuration
    llm: LlamaCpp
    rewritten_question: Optional[str]

    # Processing logic
    is_relevant: Optional[bool]
    from_memory: Optional[bool]
    chunk_responses: str

    # Output
    answer: Optional[str]

    # Memory and conversation
    memory: SimpleKVMemory
    messages: List[Dict[str, Any]]  # Full conversation history with the LLM