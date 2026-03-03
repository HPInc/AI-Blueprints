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
    history: str = "[]"  # JSON array of prior {role, content} messages for multi-turn memory


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
        self.model_path = os.path.join(model_path, "zephyr-7b-beta.Q5_K_M.gguf")
        self.secrets = secrets

        self._load_llm()

    def _load_llm(self) -> None:
        """
        Load the quantized Zephyr LLM from the .gguf file.

        Zephyr 7B Beta uses the same GGUF format as other LlamaCpp-compatible models,
        so it loads identically. The key difference is the prompt template (see _infer).

        LlamaCpp settings are optimised for NVIDIA Blackwell GPUs (GB200 / RTX 5000):
            n_gpu_layers=-1 offloads all layers to GPU for maximum speed.
            f16_kv=True uses float16 for the key-value cache to reduce VRAM usage.
            temperature=0.7 gives varied conversational responses (unlike 0.0 for RAG).
        """
        import multiprocessing
        from langchain_community.llms import LlamaCpp

        # Pydantic v2 compatibility fix — applied by every other blueprint in this repo.
        if hasattr(LlamaCpp, "model_rebuild"):
            LlamaCpp.model_rebuild()

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
                "   Run project-setup.ipynb Cell 8 to download the Zephyr 7B Beta model."
            )
            self.llm = None
            return

        logger.info(f"Loading Zephyr 7B Beta LLM: {self.model_path}")
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
            temperature=0.7,  # Conversational temperature (0.7 = natural variation)
            repeat_penalty=1.1,  # Light repetition penalty for natural chat
            streaming=False,  # Return complete response
            seed=42,  # Reproducible results
            num_threads=multiprocessing.cpu_count(),  # Use all CPU cores
            verbose=False,  # Suppress llama.cpp internal logs
        )
        logger.info("✅ Zephyr 7B Beta LLM loaded successfully")

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
        """Generate a single LLM response using the Zephyr chat template.

        Zephyr 7B Beta uses the ChatML-based prompt format:

            <|system|>
            {system_prompt}</s>
            <|user|>
            {user_message}</s>
            <|assistant|>

        The </s> token acts as the end-of-turn signal. The template ends with
        <|assistant|> without </s> so the model starts generating immediately.

        Learn more about Zephyr prompting:
            https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
        """
        if not self.llm:
            return {
                "answer": "❌ LLM not loaded. Check model_path in configs/chatbot.yaml.",
                "messages": json.dumps([]),
            }

        system_prompt = (
            inp.system_prompt or "You are a helpful and friendly AI assistant."
        )
        question = inp.question or "Hello!"

        # Parse conversation history for multi-turn memory
        try:
            prior_messages = json.loads(inp.history or "[]")
            if not isinstance(prior_messages, list):
                prior_messages = []
        except (json.JSONDecodeError, ValueError):
            prior_messages = []

        logger.info(
            f"Chatbot: processing question ({len(question)} chars) "
            f"with {len(prior_messages)} prior messages"
        )
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        # Build the Zephyr 7B Beta chat template with full conversation history.
        # Format:
        #   <|system|>{system_prompt}</s>
        #   <|user|>{msg}</s>
        #   <|assistant|>{reply}</s>
        #   ... (repeated for history turns)
        #   <|user|>{current question}</s>
        #   <|assistant|>           ← model generates from here
        prompt_parts = [f"<|system|>\n{system_prompt}</s>\n"]
        for msg in prior_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"<|user|>\n{content}</s>\n")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}</s>\n")
        prompt_parts.append(f"<|user|>\n{question}</s>\n<|assistant|>\n")
        full_prompt = "".join(prompt_parts)

        answer = self.llm.invoke(full_prompt)

        # Strip any trailing </s> that some Zephyr variants append to the response
        answer = answer.strip().rstrip("</s>").strip()

        # Build the updated messages list (history + new turn)
        messages = [
            *prior_messages,
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return {
            "answer": answer,
            "messages": json.dumps(messages, indent=2),
        }
