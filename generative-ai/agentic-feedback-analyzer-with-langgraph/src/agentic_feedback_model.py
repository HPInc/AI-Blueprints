import mlflow
import mlflow.pyfunc
from typing import Any, Dict, List
from pathlib import Path
import os
import json
import multiprocessing
import logging
from pathlib import Path
from datetime import datetime
import time
from langchain.docstore.document import Document


from langchain_community.llms import LlamaCpp
from langgraph.graph import StateGraph

from src.simple_kv_memory import SimpleKVMemory
from src.agentic_workflow import build_agentic_graph
from src.utils import logger

from pydantic import BaseModel

class AgenticModelInput(BaseModel):
    topic: str
    question: str
    input_text: str

class AgenticModelOutput(BaseModel):
    answer: str
    messages: str  # Serialized JSON string



class AgenticFeedbackModel(mlflow.pyfunc.PythonModel):
    """
    A registered MLflow PyFunc model for running the full agentic feedback analysis workflow.
    It uses a LangGraph pipeline built on top of LLaMA.cpp with in-memory caching.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext):
        """
        Load and initialize LLM and memory for use across predictions.
        """
        self.model_path: str = context.artifacts["model_path"]
        self.memory_path: str = context.artifacts["memory_path"]

        self.memory = SimpleKVMemory(Path(self.memory_path))

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

        self.graph = build_agentic_graph()
        self.compiled_graph = self.graph.compile()


    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: List[AgenticModelInput]
) -> List[AgenticModelOutput]:
        """
        Run the agentic workflow using user-provided topic, question, and input text.
        """
        results = []
        
        for row in model_input:
            topic = row.topic
            question = row.question
            input_text = row.input_text
        
            docs = [Document(page_content=input_text)]
        
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
        
            results.append(AgenticModelOutput(
                answer=final_state["answer"],
                messages=json.dumps(final_state["messages"], indent=4)
                ))
        
        return results

    @staticmethod
    def log_model(
        model_name: str,
        model_path: str,
        model_artifacts: Dict[str, str],
    ) -> None:
        """
        Log and register the model with MLflow.
        """

        mlflow.pyfunc.log_model(
            artifact_path=model_path,
            python_model=AgenticFeedbackModel(),
            artifacts=model_artifacts,
            registered_model_name=model_name,
            pip_requirements="../requirements.txt",
            code_paths=["../src"],
        )
