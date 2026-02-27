"""
VoiceModel — Business Logic Layer for Voice Assistant.

Architecture (v2.0.0):
    Focused on one capability: speech-to-text transcription + LLM response + text-to-speech.
    Implements a three-stage pipeline:
        1. Whisper (GGUF, pywhispercpp) → Transcribe audio bytes into text
        2. LlamaCpp (Llama 3.1 8B)      → Process transcribed text through the LLM
        3. XTTS v2 (CoquiTTS)           → Synthesise LLM response as audio

    Zero MLflow imports — pure Python + whisper.cpp + LangChain + CoquiTTS business logic.

What is pywhispercpp?
    pywhispercpp is a Python binding for whisper.cpp — the highly optimised C++ port
    of OpenAI's Whisper speech recognition model. It loads Whisper in GGUF format,
    enabling GPU-accelerated transcription with a much smaller memory footprint than
    the original PyTorch implementation.

    Key advantage: the GGUF model is a single file vs. the multi-file HuggingFace snapshot.

What is XTTS v2?
    XTTS (Cross-lingual Text-to-Speech) v2 by Coqui is a zero-shot TTS model that can
    clone any voice from a short audio sample (3-10 seconds). It supports 17 languages
    and produces natural-sounding speech without fine-tuning.

    In this blueprint, XTTS v2 synthesises the LLM's text response into audio so the
    voice assistant can "speak" its answer back to the user.

Pipeline:
    Audio bytes (WAV/MP3/etc.)
        ↓  base64-decode
    Raw audio bytes
        ↓  write to temp .wav file (pywhispercpp needs a file path)
    WhisperCpp(gguf_path).transcribe(wav_path)
        ↓
    Transcription text
        ↓  LlamaCpp (Llama 3.1 8B Q6_K_L)
    LLM response text
        ↓  XTTS v2 (CoquiTTS)
    Response audio (base64 WAV)

Input schema (focused):
    question      (str) — Text fallback when no audio is provided
    audio_base64  (str) — Base64-encoded audio bytes (WAV, MP3, OGG, FLAC)

Output schema:
    answer         (str) — "[Transcription: ...]\\n\\n[Response: ...]"
    messages       (str) — JSON-serialized pipeline messages
    response_audio (str) — Base64-encoded WAV of the spoken response (empty if TTS disabled)
"""

import json
import logging
import os
import tempfile
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

        self.stt_model_path = config.get("stt_model_path", "")
        self.tts_model_path = config.get("tts_model_path", "")
        self._tts_pipeline = None  # Lazy-loaded on first TTS call

        self._load_llm()

    def _load_llm(self) -> None:
        """Load the quantized LLM (Blackwell GPU-optimized settings)."""
        import multiprocessing
        from langchain_community.llms import LlamaCpp

        # Pydantic v2 compatibility fix — applied by every other blueprint in this repo.
        if hasattr(LlamaCpp, "model_rebuild"):
            LlamaCpp.model_rebuild()

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

    def _load_tts(self):
        """Lazily load the XTTS v2 TTS pipeline via CoquiTTS."""
        if self._tts_pipeline is not None:
            return self._tts_pipeline

        try:
            import torch
            from TTS.api import TTS

            tts_dir = os.path.dirname(self.tts_model_path)
            config_path = os.path.join(tts_dir, "config.json")

            if os.path.exists(config_path):
                # Load fully from the local GGUF directory
                self._tts_pipeline = TTS(
                    model_path=tts_dir,
                    config_path=config_path,
                    progress_bar=False,
                ).to("cuda" if torch.cuda.is_available() else "cpu")
            else:
                # Fall back to the CoquiTTS built-in XTTS v2 weights
                logger.warning(
                    "⚠️ XTTS config.json not found at %s — falling back to built-in XTTS v2",
                    tts_dir,
                )
                self._tts_pipeline = TTS(
                    "tts_models/multilingual/multi-dataset/xtts_v2",
                    progress_bar=False,
                ).to("cuda" if torch.cuda.is_available() else "cpu")

            logger.info("✅ XTTS v2 TTS pipeline loaded")
            return self._tts_pipeline

        except Exception as e:
            logger.warning("⚠️ TTS not loaded: %s", e)
            return None

    def _synthesize_speech(self, text: str) -> str:
        """Synthesize speech from text using XTTS v2. Returns base64-encoded WAV."""
        import base64

        tts = self._load_tts()
        if tts is None:
            return ""

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                # Cap length for educational demos; pick first built-in speaker if available
                kwargs: dict = {
                    "text": text[:500],
                    "language": "en",
                    "file_path": tmp_path,
                }
                speakers = getattr(tts, "speakers", None)
                if speakers:
                    kwargs["speaker"] = speakers[0]

                tts.tts_to_file(**kwargs)

                with open(tmp_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.warning("⚠️ TTS synthesis failed: %s", e)
            return ""

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
            "response_audio": self._synthesize_speech(response),
        }

    def _handle_audio(self, inp: VoiceInput) -> dict:
        """Transcribe audio with Whisper GGUF (pywhispercpp), then generate LLM response."""
        import base64
        from src.utils import get_response_from_llm

        # ── 1. Whisper GGUF transcription ─────────────────────────────────────
        transcription = ""
        whisper_error = None

        if os.path.exists(self.stt_model_path):
            try:
                from pywhispercpp.model import Model as WhisperCppModel

                audio_bytes = base64.b64decode(inp.audio_base64)

                # pywhispercpp requires a file path — write to a temp WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                # Normalize audio to 16 kHz mono WAV — required by Whisper (Spec 4.2.4)
                try:
                    import torchaudio

                    waveform, sample_rate = torchaudio.load(tmp_path)
                    if waveform.shape[0] > 1:  # Stereo → mono
                        waveform = waveform.mean(dim=0, keepdim=True)
                    if sample_rate != 16000:  # Resample to 16 kHz
                        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                        waveform = resampler(waveform)
                    torchaudio.save(tmp_path, waveform, 16000)
                    logger.info("✅ Audio normalized to 16 kHz mono via torchaudio")
                except Exception as ta_err:
                    logger.warning(
                        "⚠️ torchaudio normalization failed (using raw audio): %s",
                        ta_err,
                    )

                try:
                    whisper_cpp = WhisperCppModel(self.stt_model_path)
                    segments = whisper_cpp.transcribe(tmp_path)
                    transcription = " ".join(
                        seg.text.strip() for seg in segments
                    ).strip()
                    logger.info("Transcription: '%s...'", transcription[:60])
                finally:
                    os.unlink(tmp_path)  # Always clean up the temp file

            except Exception as e:
                whisper_error = str(e)
                logger.warning("⚠️ Whisper transcription failed: %s", e)
        else:
            whisper_error = f"Whisper GGUF not found at {self.stt_model_path}"
            logger.warning("⚠️ %s", whisper_error)

        if not transcription:
            return {
                "answer": (
                    f"❌ Transcription failed: {whisper_error}\n"
                    "Run project-setup.ipynb Cell 8 to download the model "
                    "into /home/jovyan/local/. See README.md → Prerequisites."
                ),
                "messages": json.dumps([]),
                "response_audio": "",
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

        # ── 3. TTS synthesis ──────────────────────────────────────────────────
        response_audio = self._synthesize_speech(response)

        messages = [
            {"role": "user", "content": f"[Voice] {transcription}"},
            {"role": "assistant", "content": response},
        ]
        return {
            "answer": f"Transcription: {transcription}\n\nResponse: {response}",
            "messages": json.dumps(messages, indent=2),
            "response_audio": response_audio,
        }
