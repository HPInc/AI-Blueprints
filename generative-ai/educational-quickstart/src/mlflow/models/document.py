"""
DocumentModel — Business Logic Layer for Document Q&A.

Architecture (v2.0.0):
    Focused on one capability: answering questions about documents.
    Implements a simplified Retrieval-Augmented Generation (RAG) pattern:
        1. Split the document into chunks
        2. Ask the LLM about each chunk independently
        3. Synthesize all chunk answers into a final unified answer

    Zero MLflow imports — pure Python + LangChain + LlamaCpp business logic.

What is Retrieval-Augmented Generation (RAG)?
    RAG grounds an LLM's answers in actual document content, reducing hallucination.
    The key insight: instead of asking "What does the document say about X?",
    you break the document into small pieces and ask the LLM about each piece,
    then combine the results. This bypasses context-window length limits.

Why chunking?
    LLMs have a maximum context window (e.g., 8192 tokens ≈ ~6000 words).
    Long documents exceed this limit. Chunking lets you process any-length document
    by asking about smaller sections and then synthesizing the answers.

Input schema (focused):
    question   (str) — The question to answer about the document
    input_text (str) — The document text to analyze

Output schema:
    answer   (str) — Synthesized answer combining all chunk analysis results
    messages (str) — JSON-serialized conversation history
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ─────── Input Schema ────────────────────────────────────────────────────────


class DocumentInput(BaseModel):
    """
    Input schema for a single document analysis request.

    The question is what you want to know about the document.
    The input_text is the document content itself (plain text).
    """

    question: str = "What are the main themes and key points of this document?"
    input_text: str = ""


# ─────── Model Class ──────────────────────────────────────────────────────────


class DocumentModel:
    """
    Document Q&A model using chunk-based analysis + LLM synthesis.

    This implements a classic RAG pattern without a vector database:
        - No embedding model needed
        - No vector store needed
        - Works on any plain text document
        - Answers are grounded in the actual document content

    For production RAG systems, you would typically add:
        - Embedding-based retrieval (find relevant chunks, not all chunks)
        - Vector database (FAISS, Chroma, Pinecone)
        - Re-ranking (score chunks by relevance before feeding to LLM)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        docs_path: Optional[str] = None,
        model_path: Optional[str] = None,
        secrets: Optional[Dict] = None,
    ):
        """
        Initialize DocumentModel.

        Constructor signature is standardized across all v2.0.0 Model classes
        so loader.py can call any of them identically.

        Args:
            config:     Configuration dict from document.yaml
            docs_path:  Path to a sample documents directory (fallback for Q&A demos)
            model_path: Full path to the .gguf LLM file in datafabric
            secrets:    Optional API keys (not used for local LLM)
        """
        self.config = config
        self.docs_path = docs_path
        self.model_path = model_path
        self.secrets = secrets

        self._load_llm()

    def _load_llm(self) -> None:
        """Load the quantized LLM from the .gguf file (Blackwell GPU-optimized settings)."""
        import multiprocessing
        from langchain_community.llms import LlamaCpp

        context_window = self.config.get("context_window", 8192)
        max_tokens = context_window // 8

        if not self.model_path:
            logger.warning(
                "⚠️ No model_path provided — DocumentModel will return errors."
            )
            self.llm = None
            return

        if not os.path.exists(self.model_path):
            logger.error(f"❌ Model file not found: {self.model_path}")
            self.llm = None
            return

        logger.info(f"Loading LLM: {self.model_path}")
        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_gpu_layers=-1,
            n_batch=512,
            n_ctx=context_window,
            max_tokens=max_tokens,
            f16_kv=True,
            use_mmap=False,
            low_vram=False,
            temperature=0.0,
            repeat_penalty=1.0,
            streaming=False,
            seed=42,
            num_threads=multiprocessing.cpu_count(),
            verbose=False,
        )
        logger.info("✅ LLM loaded successfully")

    def predict(self, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """
        Analyze documents and answer questions for each row in model_input.

        Process per row:
            1. Parse question + input_text from DataFrame row
            2. Split document into ~20-line chunks
            3. Ask LLM about each chunk: "Based on this excerpt, answer: {question}"
            4. Collect relevant chunk answers (skip "Not covered" responses)
            5. Synthesize a final answer from all relevant chunks

        Args:
            model_input: DataFrame with columns: question, input_text
            params:      Unused — present for MLflow pyfunc compatibility

        Returns:
            DataFrame with columns: answer (str), messages (str)
        """
        results = []

        for _, row in model_input.iterrows():
            try:
                inp = DocumentInput(
                    **{
                        k: str(v)
                        for k, v in row.items()
                        if v is not None and str(v).strip()
                    }
                )
            except Exception:
                inp = DocumentInput(question=str(row.get("question", "")))

            results.append(self._analyze(inp))

        return pd.DataFrame(results)

    def _analyze(self, inp: DocumentInput) -> dict:
        """Run chunk-based RAG analysis for one DocumentInput."""
        from src.utils import get_response_from_llm

        if not self.llm:
            return {
                "answer": "❌ LLM not loaded. Check model_path in configs/document.yaml.",
                "messages": json.dumps([]),
            }

        question = inp.question or "What is this document about?"
        text = inp.input_text or ""

        # Fallback: load a sample document from the docs directory if no input_text provided
        if not text and self.docs_path and os.path.exists(self.docs_path):
            sample_files = list(Path(self.docs_path).glob("*.txt"))
            if sample_files:
                with open(sample_files[0], "r", encoding="utf-8") as f:
                    text = f.read()
                logger.info(f"Using sample document: {sample_files[0].name}")

        if not text:
            return {
                "answer": "❌ No document text provided. Pass text in the 'input_text' field.",
                "messages": json.dumps([]),
            }

        # ── Chunking: split document into groups of ~20 lines ────────────────
        # Why 20 lines? It fits comfortably within the LLM context window while
        # providing enough context for meaningful analysis.
        lines = text.split("\n")
        chunk_size = 20
        chunks = [
            "\n".join(lines[i : i + chunk_size])
            for i in range(0, len(lines), chunk_size)
        ]
        chunks = [c.strip() for c in chunks if c.strip()]

        logger.info(f"Document: {len(chunks)} chunks | Question: '{question[:60]}...'")

        # ── Per-chunk Q&A ─────────────────────────────────────────────────────
        ANALYST_SYSTEM = "You are a precise document analyst."
        chunk_answers = []

        for i, chunk in enumerate(chunks[:10]):  # Cap at 10 chunks to prevent timeout
            chunk_prompt = (
                f"Document excerpt (section {i+1}):\n{chunk}\n\n"
                f"Question: {question}\n\n"
                "Answer based ONLY on the excerpt above. "
                "If this section does not contain relevant information, "
                "respond with exactly: 'Not covered in this section.'"
            )
            try:
                ans = get_response_from_llm(self.llm, ANALYST_SYSTEM, chunk_prompt)
                chunk_answers.append(ans)
            except Exception as e:
                chunk_answers.append(f"[Section {i+1} error: {str(e)}]")

        # ── Synthesis: combine chunk answers into a final response ────────────
        relevant = [ans for ans in chunk_answers if "Not covered" not in ans]

        if relevant:
            synthesis_context = "\n".join(
                f"Section {i+1}: {ans}" for i, ans in enumerate(relevant)
            )
            synthesis_prompt = (
                f"I analyzed a document in sections. Below are the per-section answers:\n\n"
                f"{synthesis_context}\n\n"
                f"Original question: {question}\n\n"
                "Write a clear, concise final answer that synthesizes the information above. "
                "Avoid repetition. Be direct and informative."
            )
            final_answer = get_response_from_llm(
                self.llm, ANALYST_SYSTEM, synthesis_prompt
            )
        else:
            final_answer = (
                "The document does not contain information relevant to this question."
            )

        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": final_answer},
        ]
        return {
            "answer": final_answer,
            "messages": json.dumps(messages, indent=2),
        }
