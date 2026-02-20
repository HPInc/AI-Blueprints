"""
VoiceModel — Business Logic Layer for Voice Assistant.

Architecture (v2.0.0):
    Focused on one capability: speech-to-text transcription + LLM response.
    Implements a two-stage pipeline:
        1. Whisper  → Transcribe audio bytes into text
        2. LlamaCpp → Process transcribed text through the LLM

    Zero MLflow imports — pure Python + Whisper + LangChain business logic.

What is Whisper?
    Whisper is OpenAI's open-source automatic speech recognition (ASR) model.
    It was trained on 680,000 hours of multilingual audio and achieves near-human
    accuracy for English transcription. We use the "large-v3" checkpoint.

    Key advantage: runs completely locally — no internet connection required.

Pipeline:
    Audio bytes (WAV/MP3/etc.)
        ↓  base64-decode
    Raw audio bytes
        ↓  write to temp file (Whisper needs a file path)
    whisper.load_model().transcribe(path)
        ↓
    Transcription text
        ↓  LlamaCpp
    LLM response text

Input schema (focused):
    question      (str) — Text fallback when no audio is provided
    audio_base64  (str) — Base64-encoded audio bytes (WAV, MP3, OGG, FLAC)

Output schema:
    answer   (str) — "[Transcription: ...]\\n\\n[Response: ...]"
    messages (str) — JSON-serialized pipeline messages
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ─────── Input Schema ────────────────────────────────────────────────────────


class VoiceInput(BaseModel):
    """
    Input schema for a single voice assistant request.

    The model accepts EITHER audio (via audio_base64) OR text (via question),
    making it testable without a microphone during development.
    """

    question: str = ""  # Text command (used when no audio is provided)
    audio_base64: str = ""  # Base64-encoded audio for Whisper transcription


# ─────── Model Class ──────────────────────────────────────────────────────────


class VoiceModel:
    """
    Voice assistant powered by Whisper (speech recognition) + LlamaCpp (LLM).

    Design principle — graceful degradation:
        If Whisper model is missing → fall back to text-only mode
        If LLM is missing           → return transcription only
        If audio is missing         → use 'question' field as text input
    """

    def __init__(
        self,
        config: Dict[str, Any],
        docs_path: Optional[str] = None,
        model_path: Optional[str] = None,
        secrets: Optional[Dict] = None,
    ):
        """
        Initialize VoiceModel.

        Constructor signature is standardized across all v2.0.0 Model classes
        so loader.py can call any of them identically.

        Args:
            config:     Configuration dict from voice.yaml
                        (reads stt_model_path for Whisper)
            docs_path:  Unused — present for loader.py compatibility
            model_path: Full path to the .gguf LLM file in datafabric
            secrets:    Optional API keys (not used for local models)
        """
        self.config = config
        self.docs_path = docs_path
        self.model_path = model_path
        self.secrets = secrets

        self.stt_model_path = config.get(
            "stt_model_path",
            "/home/jovyan/datafabric/whisper-large-v3",
        )

        self._load_llm()

    def _load_llm(self) -> None:
        """Load the quantized LLM (Blackwell GPU-optimized settings)."""
        import multiprocessing
        from langchain_community.llms import LlamaCpp

        context_window = self.config.get("context_window", 8192)
        max_tokens = context_window // 8

        if not self.model_path:
            logger.warning(
                "⚠️ No model_path provided — LLM responses will be unavailable."
            )
            self.llm = None
            return

        if not os.path.exists(self.model_path):
            logger.error(f"❌ LLM not found: {self.model_path}")
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
        Process voice or text commands for each row in model_input.

        Routing logic:
            - Row has audio_base64 → Whisper transcription + LLM response
            - Row has question only → Direct LLM response (text mode)

        Args:
            model_input: DataFrame with columns: question, audio_base64
            params:      Unused — present for MLflow pyfunc compatibility

        Returns:
            DataFrame with columns: answer (str), messages (str)
        """
        results = []

        for _, row in model_input.iterrows():
            try:
                inp = VoiceInput(
                    **{
                        k: str(v)
                        for k, v in row.items()
                        if v is not None and str(v).strip()
                    }
                )
            except Exception:
                inp = VoiceInput(question=str(row.get("question", "")))

            results.append(self._process(inp))

        return pd.DataFrame(results)

    def _process(self, inp: VoiceInput) -> dict:
        """Route the request: transcribe audio OR handle text command."""
        if inp.audio_base64:
            return self._handle_audio(inp)
        return self._handle_text(inp)

    def _handle_text(self, inp: VoiceInput) -> dict:
        """Process a text command directly (no transcription needed)."""
        from src.utils import get_response_from_llm

        command = inp.question or inp.audio_base64 or "Hello!"

        if self.llm:
            response = get_response_from_llm(
                self.llm,
                "You are a helpful voice assistant. Respond clearly and concisely.",
                command,
            )
        else:
            response = "❌ LLM not loaded. Check model_path in configs/voice.yaml."

        return {
            "answer": f"[Text input]\nCommand: {command}\nResponse: {response}",
            "messages": json.dumps(
                [
                    {"role": "user", "content": command},
                    {"role": "assistant", "content": response},
                ]
            ),
        }

    def _handle_audio(self, inp: VoiceInput) -> dict:
        """Transcribe audio with Whisper, then generate LLM response."""
        import base64
        from src.utils import get_response_from_llm

        # ── 1. Whisper transcription ──────────────────────────────────────────
        transcription = ""
        whisper_error = None

        if os.path.exists(self.stt_model_path):
            try:
                import whisper

                audio_bytes = base64.b64decode(inp.audio_base64)

                # Whisper requires a file path — write to a temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                try:
                    # Load from local .pt file if present; otherwise use model name
                    pt_files = list(Path(self.stt_model_path).glob("*.pt"))
                    model_ref = str(pt_files[0]) if pt_files else "large-v3"
                    whisper_model = whisper.load_model(model_ref)
                    transcription = (
                        whisper_model.transcribe(tmp_path).get("text", "").strip()
                    )
                    logger.info(f"Transcription: '{transcription[:60]}...'")
                finally:
                    os.unlink(tmp_path)  # Always clean up the temp file

            except Exception as e:
                whisper_error = str(e)
                logger.warning(f"⚠️ Whisper transcription failed: {e}")
        else:
            whisper_error = f"Whisper model not found at {self.stt_model_path}"
            logger.warning(f"⚠️ {whisper_error}")

        if not transcription:
            return {
                "answer": (
                    f"❌ Transcription failed: {whisper_error}\n"
                    "Download 'whisper-large-v3' into datafabric. See README.md → Prerequisites."
                ),
                "messages": json.dumps([]),
            }

        # ── 2. LLM response ───────────────────────────────────────────────────
        if self.llm:
            response = get_response_from_llm(
                self.llm,
                "You are a helpful voice assistant. Respond clearly and concisely.",
                transcription,
            )
        else:
            response = "❌ LLM not loaded. Check model_path in configs/voice.yaml."

        messages = [
            {"role": "user", "content": f"[Voice] {transcription}"},
            {"role": "assistant", "content": response},
        ]
        return {
            "answer": f"Transcription: {transcription}\n\nResponse: {response}",
            "messages": json.dumps(messages, indent=2),
        }
