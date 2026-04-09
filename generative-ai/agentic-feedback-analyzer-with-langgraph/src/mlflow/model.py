"""
Standalone Model class.

Business Logic Layer
- Handles agentic feedback analysis using LangGraph workflows
- Manages LLM initialization, memory, and multi-step agent pipeline
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

import json
import logging
import multiprocessing
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from langchain.docstore.document import Document
from langchain_community.llms import LlamaCpp
from langgraph.graph import StateGraph
from pydantic import BaseModel

from src.agentic_workflow import build_agentic_graph
from src.simple_kv_memory import SimpleKVMemory

# Set up logger
logger = logging.getLogger(__name__)


class ModelInput(BaseModel):
    """Input model for agentic feedback analysis."""

    topic: str
    question: str
    input_text: str


class ModelOutput(BaseModel):
    """Output model for agentic feedback analysis."""

    answer: str
    messages: str  # Serialized JSON string


class Model:
    """
    Agentic feedback analyzer model using LangGraph.
    Pure domain functionality with zero MLflow dependencies.
    """

    def __init__(self, config, docs_path=None, model_path=None, secrets=None):
        """
        Initialize the Model with configuration and paths.
        Constructor signature follows PR #208 pattern.

        Args:
            config: Configuration dictionary
            docs_path: Path to documents directory
            model_path: Path to LLM model file
            secrets: Dictionary containing secrets (optional, not used in this blueprint)
        """
        self.config = config
        self.docs_path = docs_path
        self.model_path = model_path
        self.secrets = secrets

        # Initialize components (extracted from original load_context)
        self._initialize_components()

    def _initialize_components(self):
        """Initialize LLM and other components."""
        import multiprocessing
        from langchain_community.llms import LlamaCpp
        from src.simple_kv_memory import SimpleKVMemory
        from src.agentic_workflow import build_agentic_graph
        from pathlib import Path

        # Initialize memory - use default path since memory_path not in constructor anymore
        memory_path = Path("../data/memory")
        self.memory = SimpleKVMemory(memory_path)
        logger.info(f"Memory initialized at: {memory_path}")

        # Initialize LLM - Use LlamaCpp directly as in the notebook
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

        logger.info(f"LlamaCpp model loaded successfully")

        # Build and compile the agentic graph (this was missing!)
        self.graph = build_agentic_graph()
        self.compiled_graph = self.graph.compile()

        # Load documents if docs_path is provided
        self.documents = []
        if self.docs_path and os.path.exists(self.docs_path):
            self._load_documents()

    def _load_documents(self):
        """Load documents from the documents directory."""
        from langchain_community.document_loaders import (
            CSVLoader,
            PyPDFLoader,
            TextLoader,
            UnstructuredExcelLoader,
            UnstructuredMarkdownLoader,
            UnstructuredWordDocumentLoader,
        )
        from pathlib import Path

        supported_extensions = {
            ".txt": TextLoader,
            ".csv": lambda path: CSVLoader(
                path, encoding="utf-8", csv_args={"delimiter": ","}
            ),
            ".xlsx": UnstructuredExcelLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".pdf": PyPDFLoader,
            ".md": UnstructuredMarkdownLoader,
        }

        for file_path in Path(self.docs_path).rglob("*"):
            if any(
                part.startswith(".") and part not in {".", ".."}
                for part in file_path.parts
            ):
                continue

            ext = file_path.suffix.lower()
            loader_class = supported_extensions.get(ext)

            if loader_class:
                try:
                    loader = loader_class(str(file_path))
                    docs = loader.load()
                    self.documents.extend(docs)
                except Exception as e:
                    print(f"Failed to load {file_path.name}: {e}")

    def predict(self, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """
        Core business logic for agentic feedback analysis.

        Args:
            model_input: DataFrame with columns: topic, question, input_text
            params: Optional parameters

        Returns:
            pandas.DataFrame with columns: answer, messages
        """
        import json
        from langchain.docstore.document import Document

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

                # Create output following original format
                result = {
                    "answer": final_state.get("answer", ""),
                    "messages": json.dumps(final_state.get("messages", []), indent=4),
                }
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "answer": f"Error processing request: {str(e)}",
                        "messages": json.dumps([], indent=4),
                    }
                )

        # Return results as DataFrame
        return pd.DataFrame(results)
