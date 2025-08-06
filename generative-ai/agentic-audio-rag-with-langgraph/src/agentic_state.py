# ─────── Standard Library Imports ───────
from typing import Any, Dict, List, Optional, TypedDict  # Type annotations for structure, collections, and optional values

# ─────── Third-Party Package Imports ───────
from langchain.docstore.document import Document  # Core document abstraction used across LangChain
from langchain_community.llms import LlamaCpp  # Interface for running Llama models locally with llama.cpp

# ─────── Local Application-Specific Imports ───────
from src.simple_kv_memory import SimpleKVMemory  # In-memory key-value store used for maintaining agent state


class AgenticState(TypedDict, total=False):
    """
    Shared state that flows through every LangGraph node in the agentic audio RAG pipeline
    """

    # Input metadata
    question: str                           # Original user question
    file_id: str                            # Document identifier/path of the media file

    # Transcript & chunking
    docs: List[Document]                    # Full transcript wrapped in one document
    chunks: List[Document]                  # Overlapping transcript chunks

    # LLM configuration & rewritten query
    llm: LlamaCpp
    rewritten_question: Optional[str]

    # Processing logic & control flags
    is_relevant: Optional[bool]             # Set by relevance filter
    from_memory: Optional[bool]             # True when served from cache

    # Intermediate results
    chunk_responses: str                    # Per-chunk answers
    snippets: List[Dict[str, str]]          # [{start,end,text},…]

    # Output
    answer: Optional[str]

    # Memory and conversation log
    memory: SimpleKVMemory
    messages: List[Dict[str, Any]]  # Full conversation history with the LLM