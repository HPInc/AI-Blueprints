"""
Standalone Model class.

Business Logic Layer
- Handles agentic GitHub repository analysis workflow using LangGraph
- Manages LLM initialization, memory, and agentic graph compilation
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

# ─────── Standard Library Imports ───────
import json  # JSON parsing and serialization
import logging  # Logging utilities
import multiprocessing  # Multi-process support for concurrency
import os  # Operating system interaction
import time  # Time-related functions
from datetime import datetime  # Date and time manipulation
from pathlib import Path  # Object-oriented filesystem paths
from typing import Any, Dict, List  # Static typing support

# ─────── Third-Party Package Imports ───────
from langchain.docstore.document import Document  # Core document abstraction for LangChain
from langchain_community.llms import LlamaCpp  # Local LLM interface for Llama.cpp
from langgraph.graph import StateGraph  # LangGraph for stateful agent workflows
from pydantic import BaseModel  # Data validation and model parsing

# ─────── Local Application-Specific Imports ───────
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.agentic_workflow import build_agentic_graph  # Custom LangGraph construction logic
from src.simple_kv_memory import SimpleKVMemory  # In-memory key-value store for agent state
from src.utils import logger  # Project-wide configured logger


class AgenticModelInput(BaseModel):
    topic: str
    question: str
    input_text: str


class AgenticModelOutput(BaseModel):
    answer: str
    messages: str  # Serialized JSON string


class Model:
    """
    Standalone model class with no MLflow inheritance.
    Handles agentic GitHub repository analysis using LangGraph workflow.
    """
    
    def __init__(self, model_path: str, memory_path: str):
        """
        Initialize the model with direct dependencies - no MLflow context.
        Extract all initialization logic from original load_context method.
        
        Args:
            model_path: Path to the local LLM model file
            memory_path: Path to the memory storage directory
        """
        self.model_path = model_path
        self.memory_path = memory_path
        
        # Initialize memory
        self.memory = SimpleKVMemory(Path(self.memory_path))
        
        # Initialize LLM with same configuration as original service
        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_gpu_layers=-1,
            n_batch=512,
            n_ctx=8192,
            max_tokens=1024,
            f16_kv=True,
            use_mmap=False,
            low_vram=False,
            rope_scaling=None,
            temperature=0.0,
            repeat_penalty=1.0,
            streaming=False,
            stop=None,
            seed=42,
            num_threads=multiprocessing.cpu_count(),
            verbose=False,
        )
        
        # Build and compile the agentic graph
        self.graph = build_agentic_graph()
        self.compiled_graph = self.graph.compile()

    def predict(self, model_input: List[AgenticModelInput], params=None) -> List[AgenticModelOutput]:
        """
        Core business logic extracted from original service predict method.
        Remove context parameter - use instance variables instead.
        Must return same data structure as original.
        
        Args:
            model_input: List of AgenticModelInput objects with topic, question, and input_text
            params: Optional parameters (maintained for API compatibility)
            
        Returns:
            List of AgenticModelOutput objects with answer and messages
        """
        results = []

        for row in model_input:
            topic = row.topic
            question = row.question
            input_text = row.input_text

            docs = [Document(page_content=input_text)]
        
            # Run the agentic workflow using compiled graph
            final_state = self.compiled_graph.invoke(
                input={
                    "topic": topic,
                    "question": question,
                    "docs": docs,
                    "memory": self.memory,
                    "llm": self.llm,  # Use self.llm instead of context
                    "messages": [],
                }
            )
        
            results.append(AgenticModelOutput(
                answer=final_state["answer"],
                messages=json.dumps(final_state["messages"], indent=4)
            ))
        
        return results
