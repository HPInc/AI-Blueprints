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
from langchain_core.documents import (
    Document,
)  # Core document abstraction for LangChain
from langchain_community.llms import LlamaCpp  # Local LLM interface for Llama.cpp
from langgraph.graph import StateGraph  # LangGraph for stateful agent workflows
from pydantic import BaseModel  # Data validation and model parsing

# ─────── Local Application-Specific Imports ───────
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.agentic_workflow import (
    build_agentic_graph,
)  # Custom LangGraph construction logic
from src.simple_kv_memory import (
    SimpleKVMemory,
)  # In-memory key-value store for agent state
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

    def __init__(self, config, docs_path=None, model_path=None, secrets=None):
        """
        Initialize the Model with configuration and paths.

        Args:
            config: Configuration dictionary with model settings
            docs_path: Path to documents directory (not used for GitHub repo analyzer)
            model_path: Path to LLM model file
            secrets: Dictionary containing secrets (optional, not used in this blueprint)
        """
        self.config = config
        self.docs_path = docs_path
        self.model_path = model_path
        self.secrets = secrets

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize LLM, memory, and agentic graph."""
        # Initialize memory - use default path
        memory_path = Path("../data/memory")
        self.memory = SimpleKVMemory(memory_path)
        logger.info(f"Memory initialized at: {memory_path}")

        # Initialize LLM
        if not self.model_path:
            raise ValueError(
                "model_path is required for LLM initialization. Please configure it in config.yaml"
            )

        logger.info(f"Initializing LLM with model_path: {self.model_path}")

        # Get context window and related configs from config
        context_window = self.config.get("context_window", 8192)
        max_tokens = context_window // 8

        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_gpu_layers=-1,
            n_batch=512,
            n_ctx=context_window,
            max_tokens=max_tokens,
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

        if self.llm is None:
            raise ValueError("Failed to initialize LLM - LlamaCpp returned None")

        logger.info("LlamaCpp model loaded successfully")

        # Build and compile the agentic graph
        self.graph = build_agentic_graph()
        self.compiled_graph = self.graph.compile()
        logger.info("Agentic graph compiled successfully")

    def predict(self, model_input, params=None):
        """
        Core business logic for agentic GitHub repository analysis.
        Supports both DataFrame and List[AgenticModelInput] for compatibility.

        Args:
            model_input: pandas.DataFrame with columns (topic, question, input_text)
                        OR List[AgenticModelInput]
            params: Optional parameters

        Returns:
            pandas.DataFrame with columns (answer, messages) OR List[AgenticModelOutput]
        """
        import pandas as pd

        # Handle DataFrame input (MLflow standard)
        if isinstance(model_input, pd.DataFrame):
            results = []

            # Process each row in the DataFrame
            for _, row in model_input.iterrows():
                topic = row.get("topic", "")
                question = row.get("question", "")
                input_text = row.get("input_text", "")

                # Create document from input text
                docs = [Document(page_content=input_text)]

                # Run the agentic workflow using compiled graph
                try:
                    final_state = self.compiled_graph.invoke(
                        input={
                            "topic": topic,
                            "question": question,
                            "docs": docs,
                            "memory": self.memory,
                            "llm": self.llm,
                            "messages": [],
                        }
                    )

                    result = {
                        "answer": final_state.get("answer", ""),
                        "messages": json.dumps(
                            final_state.get("messages", []), indent=4
                        ),
                    }
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing request: {str(e)}")
                    results.append(
                        {
                            "answer": f"Error processing request: {str(e)}",
                            "messages": json.dumps([], indent=4),
                        }
                    )

            # Return results as DataFrame
            return pd.DataFrame(results)

        # Handle List[AgenticModelInput] (for notebook compatibility)
        else:
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
                        "llm": self.llm,
                        "messages": [],
                    }
                )

                results.append(
                    AgenticModelOutput(
                        answer=final_state["answer"],
                        messages=json.dumps(final_state["messages"], indent=4),
                    )
                )

            return results
