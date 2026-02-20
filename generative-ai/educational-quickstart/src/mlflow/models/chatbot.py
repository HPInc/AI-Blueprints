"""
ChatbotModel — Business Logic Layer for Conversational AI.

Architecture (v2.0.0):
    This class is the Business Logic Layer for the chatbot capability.
    It has ZERO MLflow imports — it is a plain Python class.

    MLflow integration is handled by:
        loader.py → instantiates this class from saved artifacts
        logger.py → packages this class's config and demo into an MLflow artifact

Learn more:
    https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html

Input schema (focused):
    question      (str) — The user's message
    system_prompt (str) — LLM persona / role instructions (optional)

Output schema:
    answer   (str) — LLM text response
    messages (str) — JSON-serialized conversation history
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ─────── Input / Output Schema ───────────────────────────────────────────────


class ChatbotInput(BaseModel):
    """
    Input schema for a single chatbot inference request.

    Why Pydantic?
        Pydantic validates that fields have the correct types and provides clear
        error messages when input is malformed — helpful in educational settings.
    """

    question: str = ""
    system_prompt: str = "You are a helpful and friendly AI assistant."


# ─────── Model Class ──────────────────────────────────────────────────────────


class ChatbotModel:
    """
    Conversational AI chatbot powered by a local LLM via llama.cpp.

    This is the SAME class that gets registered in the MLflow Model Registry.
    When you run model.predict() in a notebook, you are calling the exact same
    inference logic that runs in production — zero divergence.

    What is llama.cpp?
        llama.cpp is an inference engine for quantized LLMs written in C++.
        It runs efficiently on both CPU and GPU without requiring PyTorch.
        We use it through the langchain_community.llms.LlamaCpp wrapper.

    GGUF format:
        The model file uses the GGUF format — a single binary file containing
        the model weights in a quantized (compressed) form that fits in less VRAM.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        docs_path: Optional[str] = None,
        model_path: Optional[str] = None,
        secrets: Optional[Dict] = None,
    ):
        """
        Initialize the ChatbotModel.

        Constructor signature is standardized across all v2.0.0 Model classes
        so that loader.py can call any of them identically.

        Args:
            config:     Configuration dict from chatbot.yaml (context_window, etc.)
            docs_path:  Unused by chatbot — present for loader.py compatibility
            model_path: Full path to the .gguf LLM file in datafabric
            secrets:    Optional API keys / tokens (not used for local LLM)
        """
        self.config = config
        self.docs_path = docs_path
        self.model_path = model_path
        self.secrets = secrets

        self._load_llm()

    def _load_llm(self) -> None:
        """
        Load the quantized LLM from the .gguf file.

        LlamaCpp settings are optimized for NVIDIA Blackwell GPUs (GB200 / RTX 5000):
            n_gpu_layers=-1 offloads all layers to GPU for maximum speed.
            f16_kv=True uses float16 for the key-value cache to reduce VRAM usage.
            temperature=0.0 makes outputs deterministic — important for reproducibility.
        """
        import multiprocessing
        from langchain_community.llms import LlamaCpp

        context_window = self.config.get("context_window", 8192)
        max_tokens = context_window // 8  # Reserve ⅛ of context for the response

        if not self.model_path:
            logger.warning(
                "⚠️ No model_path provided — LLM will be unavailable.\n"
                "   Set model_path in configs/chatbot.yaml or pass it to ChatbotModel()."
            )
            self.llm = None
            return

        if not os.path.exists(self.model_path):
            logger.error(
                f"❌ Model file not found at: {self.model_path}\n"
                "   Download the GGUF model into datafabric. See README.md → Prerequisites."
            )
            self.llm = None
            return

        logger.info(f"Loading LLM: {self.model_path}")
        logger.info(
            f"Context: {context_window} tokens | Max response: {max_tokens} tokens"
        )

        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_gpu_layers=-1,  # Offload ALL layers to GPU
            n_batch=512,  # Tokens processed per batch
            n_ctx=context_window,  # Total context window size
            max_tokens=max_tokens,  # Max tokens in the response
            f16_kv=True,  # Float16 KV-cache (saves VRAM)
            use_mmap=False,  # Disable memory-mapped file I/O
            low_vram=False,  # Full VRAM mode (Blackwell default)
            temperature=0.0,  # Deterministic outputs
            repeat_penalty=1.0,  # No repetition penalty
            streaming=False,  # Return complete response
            seed=42,  # Reproducible results
            num_threads=multiprocessing.cpu_count(),  # Use all CPU cores
            verbose=False,  # Suppress llama.cpp internal logs
        )
        logger.info("✅ LLM loaded successfully")

    def predict(self, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """
        Run chatbot inference on each row of model_input.

        This method is called by:
            1. Notebooks directly:
               model.predict(pd.DataFrame([{"question": "...", "system_prompt": "..."}]))
            2. MLflow at serving time:
               MLflow wraps this method and exposes it via HTTP POST /invocations

        Why a DataFrame?
            MLflow's pyfunc interface requires DataFrames so that batch inference
            (multiple questions in one call) works seamlessly.

        Args:
            model_input: DataFrame with columns: question, system_prompt
            params:      Unused — present for MLflow pyfunc interface compatibility

        Returns:
            DataFrame with columns: answer (str), messages (str)
        """
        results = []

        for _, row in model_input.iterrows():
            try:
                inp = ChatbotInput(
                    **{
                        k: str(v)
                        for k, v in row.items()
                        if v is not None and str(v).strip()
                    }
                )
            except Exception:
                inp = ChatbotInput(question=str(row.get("question", "")))

            results.append(self._infer(inp))

        return pd.DataFrame(results)

    def _infer(self, inp: ChatbotInput) -> dict:
        """Generate a single LLM response for one ChatbotInput."""
        from src.utils import get_response_from_llm

        if not self.llm:
            return {
                "answer": "❌ LLM not loaded. Check model_path in configs/chatbot.yaml.",
                "messages": json.dumps([]),
            }

        system_prompt = (
            inp.system_prompt or "You are a helpful and friendly AI assistant."
        )
        question = inp.question or "Hello!"

        logger.info(f"Chatbot: processing question ({len(question)} chars)")
        answer = get_response_from_llm(self.llm, system_prompt, question)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return {
            "answer": answer,
            "messages": json.dumps(messages, indent=2),
        }
