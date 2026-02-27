"""
VoiceModel — Business Logic Layer for Voice Assistant.

Architecture (v2.0.0):
    Focused on one capability: speech-to-text transcription + LLM response + text-to-speech.
    Implements a three-stage pipeline:
        1. Whisper (transformers pipeline) → Transcribe audio bytes into text
        2. LlamaCpp (Llama 3.1 8B)        → Process transcribed text through the LLM
        3. XTTS v2 (CoquiTTS)             → Synthesise LLM response as audio

    Zero MLflow imports — pure Python + transformers + LangChain + CoquiTTS business logic.

What is the Whisper transformers pipeline?
    Uses `transformers.pipeline("automatic-speech-recognition")` with the official
    openai/whisper-large-v3-turbo HuggingFace model (safetensors snapshot).
    Pure Python/PyTorch — all errors are catchable Python exceptions (no C-level segfaults).
    Accepts a float32 numpy array (16 kHz mono) directly, so no intermediate file is needed
    beyond the initial base64 decode + librosa normalization step.

What is XTTS v2?
    XTTS (Cross-lingual Text-to-Speech) v2 by Coqui is a zero-shot TTS model that can
    clone any voice from a short audio sample (3-10 seconds). It supports 17 languages
    and produces natural-sounding speech without fine-tuning.

    In this blueprint, XTTS v2 synthesises the LLM's text response into audio so the
    voice assistant can "speak" its answer back to the user.

Pipeline:
    Audio bytes (WAV/MP3/etc.)
        ↓  base64-decode + librosa normalize (16 kHz mono)
    float32 audio array
        ↓  transformers.pipeline("automatic-speech-recognition")
        ↓
    Transcription text
        ↓  LlamaCpp (Llama 3.1 8B Q6_K_L)
    LLM response text
        ↓  XTTS v2 (CoquiTTS)
    Response audio (base64 WAV)

Input schema:
    question      (str) — Not used; present for schema backward-compatibility only.
    audio_base64  (str) — Base64-encoded audio bytes (WAV, MP3, OGG, FLAC); required.

Output schema:
    answer         (str) — "Transcription: ...\\n\\nResponse: ..."
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

    audio_base64 (base64-encoded WAV/MP3/OGG/FLAC) is required.
    question is preserved in the schema for backward-compatibility but is not used.
    """

    question: str = ""  # Text command (used when no audio is provided)
    audio_base64: str = ""  # Base64-encoded audio for Whisper transcription


# ─────── Model Class ──────────────────────────────────────────────────────────


class VoiceModel:
    """
    Voice assistant powered by Whisper (speech recognition) + LlamaCpp (LLM) + XTTS v2 (TTS).

    Accepts audio input only. The three-stage pipeline is:
        1. Whisper (transformers pipeline) → transcribe audio to text
        2. LlamaCpp (Llama 3.1 8B)        → generate a response
        3. XTTS v2 (CoquiTTS)             → synthesise the response as speech

    Design principle — graceful degradation:
        If Whisper model is missing → error with instructions to run project-setup
        If LLM is missing           → return transcription only
        If TTS model is missing     → return text response with empty response_audio
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
        """Route the request to the audio pipeline; audio_base64 is required."""
        if not inp.audio_base64:
            return {
                "answer": (
                    "❌ No audio provided. Pass base64-encoded audio in the "
                    "audio_base64 field (WAV, MP3, OGG, or FLAC)."
                ),
                "messages": json.dumps([]),
                "response_audio": "",
            }
        return self._handle_audio(inp)

    def _load_tts(self):
        """Lazily load the XTTS v2 TTS pipeline via CoquiTTS.

        Loading strategy:
            1. If tts_model_path points to a directory containing config.json,
               load from that local path.  This works in both development
               (/home/jovyan/local/xtts-v2/) and at MLflow serve time
               (<artifact_models>/xtts-v2/) because loader.py resolves the
               path from the bundled artifact before constructing this object.
            2. Fall back to the CoquiTTS registry model name so that the
               assistant is still functional even if the artifact was not
               copied (CoquiTTS will download to ~/.local/share/tts/).

        VRAM management (three models share ~8 GB VRAM):
            • Pre-check: free VRAM is queried via torch.cuda.mem_get_info() before any
              tensor is placed. Device (CUDA vs CPU) is decided once, deterministically.
            • float16: synthesiser cast to fp16 unconditionally on CUDA (~1.8 GB → ~0.9 GB).
        """
        if self._tts_pipeline is not None:
            return self._tts_pipeline

        try:
            import torch
            from TTS.api import TTS

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # tts_model_path is a directory (e.g. /home/jovyan/local/xtts-v2)
            # containing config.json and model.pth from coqui/XTTS-v2.
            tts_dir = self.tts_model_path
            config_path = os.path.join(tts_dir, "config.json") if tts_dir else ""

            if tts_dir and os.path.isdir(tts_dir) and os.path.exists(config_path):
                tts_obj = TTS(
                    model_path=tts_dir,
                    config_path=config_path,
                    progress_bar=False,
                )
                source_desc = f"local path: {tts_dir}"
            else:
                # Fall back to the CoquiTTS built-in registry download
                logger.warning(
                    "⚠️ XTTS model directory not found at %s — "
                    "falling back to CoquiTTS registry download (~1.8 GB)",
                    tts_dir or "(not set)",
                )
                tts_obj = TTS(
                    "tts_models/multilingual/multi-dataset/xtts_v2",
                    progress_bar=False,
                )
                source_desc = "CoquiTTS registry"

            # ── Determine target device before any allocation ─────────────────────
            # Pre-check free VRAM using the CUDA allocator's own accounting so the
            # device decision is made once, deterministically, before any tensor is
            # placed.  XTTS v2 fp16 peak ≈ 0.9 GB; add a 256 MB safety margin.
            _XTTS_VRAM_REQUIRED = int(1.15 * 1024**3)  # 1.15 GB in bytes
            if device == "cuda":
                free_vram, _ = torch.cuda.mem_get_info()
                if free_vram < _XTTS_VRAM_REQUIRED:
                    logger.warning(
                        "⚠️ Insufficient free VRAM for XTTS (%d MB free, %d MB required)"
                        " — loading on CPU",
                        free_vram // (1024**2),
                        _XTTS_VRAM_REQUIRED // (1024**2),
                    )
                    device = "cpu"

            tts_obj.to(device)

            # ── Cast synthesiser to float16 on CUDA ───────────────────────────────
            # XTTS v2 always exposes synthesizer.tts_model and supports fp16
            # inference.  Halves VRAM from ~1.8 GB to ~0.9 GB with no quality loss.
            if device == "cuda":
                tts_obj.synthesizer.tts_model = tts_obj.synthesizer.tts_model.half()
                logger.info("✅ XTTS v2 cast to float16 (~0.9 GB VRAM)")

            self._tts_pipeline = tts_obj
            logger.info("✅ XTTS v2 loaded from %s on %s", source_desc, device.upper())
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

    def _handle_audio(self, inp: VoiceInput) -> dict:
        """Transcribe audio with Whisper (transformers pipeline), then generate LLM response."""
        import base64
        import gc
        import torch
        from src.utils import get_response_from_llm

        # ── 1. Whisper transcription (transformers pipeline) ──────────────────
        transcription = ""
        whisper_error = None
        stt = None

        if os.path.isdir(self.stt_model_path):
            try:
                import librosa
                from transformers import pipeline as hf_pipeline

                audio_bytes = base64.b64decode(inp.audio_base64)

                # Write raw audio to temp file, then normalize to 16 kHz mono
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                try:
                    audio_data, _ = librosa.load(tmp_path, sr=16000, mono=True)
                    logger.info("✅ Audio normalized to 16 kHz mono via librosa")
                finally:
                    os.unlink(tmp_path)

                device = (
                    0 if torch.cuda.is_available() else -1
                )  # 0 = first GPU, -1 = CPU
                stt = hf_pipeline(
                    "automatic-speech-recognition",
                    model=self.stt_model_path,
                    device=device,
                    torch_dtype=(
                        torch.float16 if torch.cuda.is_available() else torch.float32
                    ),
                )
                result = stt(audio_data, generate_kwargs={"language": "english"})
                transcription = result["text"].strip()
                logger.info("Transcription: '%s...'", transcription[:60])

            except Exception as e:
                whisper_error = str(e)
                logger.warning("⚠️ Whisper transcription failed: %s", e)
            finally:
                # ── Release Whisper VRAM before LLM inference + XTTS load ────────────
                # Without explicit cleanup, the Whisper model stays pinned in GPU RAM.
                # Peak with all three models live simultaneously:
                #   LLM ~6.6 GB + Whisper ~1.2 GB + XTTS ~0.9 GB ≈ 8.7 GB
                # Free Whisper before XTTS loads to stay within the VRAM budget.
                if stt is not None:
                    del stt
                    stt = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("✅ Whisper VRAM released")
        else:
            whisper_error = f"Whisper model directory not found: {self.stt_model_path}"
            logger.warning("⚠️ %s", whisper_error)

        if not transcription:
            return {
                "answer": (
                    f"❌ Transcription failed: {whisper_error}\n"
                    "Run project-setup.ipynb Cell 9 to download the Whisper model "
                    "into /home/jovyan/local/whisper-large-v3-turbo/. "
                    "See README.md → Prerequisites."
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
