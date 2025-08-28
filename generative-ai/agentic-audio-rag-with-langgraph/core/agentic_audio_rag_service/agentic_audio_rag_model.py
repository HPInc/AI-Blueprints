"""
Standalone AgenticAudioModel class.

Business Logic Layer
- Manages model initialization, embeddings, vector database, and prediction logic
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

# ─────── Standard Library Imports ───────
from __future__ import annotations # Future-proofing for type annotations
import json  # JSON parsing and serialization
import re # Regular expressions
import logging  # Logging utilities
import os  # Operating system interaction
from pathlib import Path  # Object-oriented filesystem paths
from typing import Any, Dict, List, Tuple, TypedDict, Annotated, Optional  # Static typing support
import numpy as np  # Numerical computing with arrays
import soundfile as sf # Audio file reading and writing
from types import SimpleNamespace # Simple object for attribute access
import operator # Functional programming utilities
import uuid
import sys
import pandas as pd

# ─────── Third-Party Package Imports ───────
from langchain.docstore.document import Document  # Core document abstraction for LangChain
from langgraph.graph import StateGraph, END  # LangGraph for stateful agent workflows
from pydantic import BaseModel  # Data validation and model parsing
import torch  # PyTorch for deep learning
import torchaudio
import faiss
from transformers import (
    ClapProcessor, ClapModel,
    Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration,
)
from qwen_omni_utils import process_mm_info

# ─────── Local Application-Specific Imports ───────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.agentic_workflow import build_audio_agentic_graph  # Custom LangGraph construction logic
from src.simple_kv_memory import SimpleKVMemory, _mem_get, _mem_put  # In-memory key-value store for agent state
from src.model_selection import ModelSelector # Model selection utility
from src.utils import setup_model_environment  # Project-wide configured logger
from src.segment_audio_embeddings import AudioIndex, ensure_wav

# Add the src directory to the path to import utilities
from src.utils import get_context_window, dynamic_retriever, format_docs_with_adaptive_context, load_secrets_to_env

# Set up logger
logger = logging.getLogger(__name__)

INDEX_VECS_NPY = "audio_vecs.npy"
INDEX_META_JSON = "audio_meta.json"
MEMORY_FILENAME = "kv_memory.json"
# Build index from MEDIA_DIR and snapshot to artifacts/
AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS

RELEVANCE_THRESHOLD = 0.18
FETCH_K = 24     # breadth for stage-1
TOP_K   = 6      # final segments

 # 
class _QwenAdapter:
    def __init__(self, proc, model):
        self.processor = proc
        self.model = model
        self.qwen_default_system = (
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
            "capable of perceiving auditory and visual inputs, as well as generating text and speech."
        )

    def _sanitize(self, txt: str) -> str:
        txt = re.sub(r"^(?:\d+\s*)?(?:Human:|User:|Assistant:|System:)\s*", "", txt, flags=re.IGNORECASE).strip()
        txt = re.sub(r"([!?.])\1{2,}", r"\1", txt)
        return txt

    def answer(self, question: str, audio_hits: list, return_audio: bool = False) -> dict:
        user_content = [{
            "type": "text",
            "text": ("Answer ONLY using the provided audio clips. "
                     "If the clips do not contain the answer, reply exactly: NOT_FOUND_IN_AUDIO.\n"
                     f"Question: {question}")
        }]

        usable = 0
        for h in (audio_hits or []):
            audio_full, sr = sf.read(h["wav_path"])
            if audio_full.ndim == 2:
                audio_full = audio_full.mean(axis=1)
            s0, s1 = int(h["start_s"] * sr), int(h["end_s"] * sr)
            s0 = max(0, min(s0, len(audio_full)))
            s1 = max(0, min(s1, len(audio_full)))
            if s1 <= s0:
                continue
            seg = audio_full[s0:s1].astype(np.float32)
            user_content.append({"type": "audio", "audio": seg, "sampling_rate": sr})
            usable += 1

        if usable == 0:
            return {"answer": "Not found in audio.", "evidence": []}

        conv = [
            {"role": "system", "content": [{"type": "text", "text": self.qwen_default_system}]},
            {"role": "user",   "content": user_content},
        ]

        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)

        audios = audios if (audios and len(audios) > 0) else None
        images = images if (images and len(images) > 0) else None
        videos = videos if (videos and len(videos) > 0) else None

        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False
        ).to(self.model.device)

        tok = getattr(self.processor, "tokenizer", None)
        eos_id = getattr(tok, "eos_token_id", None)
        try:
            chat_eos = tok.convert_tokens_to_ids("<|im_end|>") if tok is not None else None
        except Exception:
            chat_eos = None
        eos_ids = [i for i in (eos_id, chat_eos) if i is not None] or None
        pad_id = getattr(tok, "pad_token_id", eos_id)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                use_audio_in_video=False,
                max_new_tokens=192,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.1,
                no_repeat_ngram_size=4,
                eos_token_id=eos_ids,
                pad_token_id=pad_id,
                return_dict_in_generate=True,
            )

        prompt_ids = inputs.get("input_ids", None)
        prompt_len = prompt_ids.shape[1] if prompt_ids is not None else 0
        seqs = getattr(out, "sequences", None)

        if seqs is None:
            answer = ""
        else:
            try:
                new_tokens = seqs[:, prompt_len:]
            except Exception:
                new_tokens = seqs
            decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=True)
            answer = (decoded[0].strip() if decoded else "").strip()

        answer = self._sanitize(answer)
        if answer.upper().startswith("NOT_FOUND_IN_AUDIO"):
            answer = "Not found in audio."

        evid = [{
            "file_name": h["file_name"], "file_path": h["file_path"],
            "start_s": h["start_s"], "end_s": h["end_s"],
            "score": h.get("score_mmr", h.get("score", 0.0)),
        } for h in (audio_hits or [])]

        return {"answer": (answer if answer else "Not found in audio."), "evidence": evid}

def _normalize_vecs(vecs: np.ndarray) -> np.ndarray:
    x = vecs.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32)

def _set_embeddings(MEDIA_DIR, index_dir, config_path):
    # Reuse the CLAP init + embedding utilities from your run-workflow notebook
    # If you already defined them earlier in this kernel, skip redefining.

    # CLAP init
    CLAP_REPO = "laion/clap-htsat-unfused"
    clap_device = "cuda" if torch.cuda.is_available() else "cpu"
    clap_processor = ClapProcessor.from_pretrained(CLAP_REPO)
    clap_model = ClapModel.from_pretrained(CLAP_REPO).to(clap_device).eval()

    try:
        clap_model.to("cpu")
        clap_device = "cpu"
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("CLAP moved to CPU; GPU cache cleared")
    except Exception as e:
        print("Skipping CLAP offload:", e)

    def _resample_to_48k(wav: np.ndarray, sr: int, target_sr: int = 48000) -> np.ndarray:
        if sr == target_sr:
            return wav.astype(np.float32, copy=False)
        try:
            t = torch.as_tensor(wav, dtype=torch.float32).unsqueeze(0)
            t48 = torchaudio.functional.resample(t, sr, target_sr)
            return t48.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception:
            x = np.linspace(0, 1, num=wav.shape[0], dtype=np.float64, endpoint=False)
            y = np.interp(np.linspace(0, 1, num=int(round(wav.shape[0] * target_sr / sr)), endpoint=False),
                        x, wav.astype(np.float64, copy=False))
            return y.astype(np.float32)

    @torch.no_grad()
    def clap_embed_audio(wav: np.ndarray, sr: int) -> np.ndarray:
        wav48 = _resample_to_48k(wav, sr, 48000)
        inp = clap_processor(audios=[wav48], sampling_rate=48000, return_tensors="pt").to(clap_device)
        out = clap_model.get_audio_features(**inp)
        vec = out.cpu().numpy()[0]
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec.astype(np.float32)

    # @torch.no_grad()
    # def clap_embed_text(query: str) -> np.ndarray:
    #     inp = clap_processor(text=[query], return_tensors="pt").to(clap_device)
    #     out = clap_model.get_text_features(**inp)
    #     vec = out.cpu().numpy()[0]
    #     vec = vec / (np.linalg.norm(vec) + 1e-12)
    #     return vec.astype(np.float32)

    # Segmentation
    def segment_audio(wav_path: str, window_s: float = 30.0, hop_s: float = 15.0) -> List[Tuple[int, int, np.ndarray, int]]:
        audio, sr = sf.read(wav_path)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        n = len(audio); win = int(window_s * sr); hop = int(hop_s * sr)
        if n == 0: return []
        segs, i = [], 0
        while i < n:
            j = min(i + win, n)
            segs.append((i, j, audio[i:j], sr))
            if j == n: break
            i += hop
        return segs

    # In-memory FAISS index shell
    class AudioIndex:
        def __init__(self, dim: int = 512):
            self.index = faiss.IndexFlatIP(dim)
            self.meta: List[Dict[str, Any]] = []
        def add(self, vecs: np.ndarray, metas: List[Dict[str, Any]]):
            # cosine via normalized IP
            vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
            self.index.add(vecs.astype(np.float32))
            self.meta.extend(metas)
        def search(self, qvec: np.ndarray, k: int = 6) -> List[Dict[str, Any]]:
            qvec = qvec.astype(np.float32)
            qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
            D, I = self.index.search(qvec[np.newaxis, :], k)
            out = []
            for idx, score in zip(I[0], D[0]):
                if 0 <= idx < len(self.meta):
                    m = dict(self.meta[idx]); m["score"] = float(score)
                    out.append(m)
            return out

    # empty initial memory

    # Collect media
    media_paths = []
    for p in sorted(Path(MEDIA_DIR).rglob("*")):
        if any(part.startswith(".") and part not in {".", ".."} for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in MEDIA_EXTS:
            media_paths.append(p)

    # Embed segments
    audio_index = AudioIndex(dim=512)
    for media_path in media_paths:
        wav_path = ensure_wav(AUDIO_EXTS, VIDEO_EXTS, str(media_path))
        segs = segment_audio(wav_path, window_s=30.0, hop_s=15.0)
        if not segs: continue
        vecs, metas = [], []
        for (s0, s1, wav_seg, sr) in segs:
            v = clap_embed_audio(wav_seg, sr); vecs.append(v)
            metas.append({
                "file_path": str(media_path),
                "file_name": media_path.name,
                "wav_path": wav_path,
                "start_s": float(s0 / sr),
                "end_s": float(s1 / sr),
            })
        audio_index.add(np.stack(vecs, axis=0), metas)

    # Persist index vectors + metadata as model artifacts
    # We need the raw (already normalized) vectors; FAISS can't be pickled easily across runtimes.
    # Re-run a pass to collect vectors in the same order FAISS used:
    # (For simplicity, we re-embed here; for large corpora, persist as you add)
    vecs = []
    for m in audio_index.meta:
        audio, sr = sf.read(m["wav_path"])
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        i0 = int(m["start_s"] * sr); i1 = int(m["end_s"] * sr)
        wav_seg = audio[i0:i1].astype(np.float32, copy=False)
        vecs.append(clap_embed_audio(wav_seg, sr))
    vecs = np.stack(vecs, axis=0).astype(np.float32)
    np.save(index_dir / INDEX_VECS_NPY, vecs)

    with open(index_dir / INDEX_META_JSON, "w") as f:
        json.dump(audio_index.meta, f, ensure_ascii=False, indent=2)

    # Write a simple runtime config
    config = {
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "fetch_k": FETCH_K,
        "top_k": TOP_K,
        "clap_repo": CLAP_REPO,
        "media_root": str(MEDIA_DIR),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("Indexed segments:", len(audio_index.meta))

    
class AgenticAudioModel:
    """
    Standalone agentic audio RAG model class with no MLflow inheritance.
    Handles RAG-based question answering with audio files.
      - CLAP for retrieval + MMR reranking
      - Qwen2.5 Omni Thinker for audio reasoning
      - LangGraph for memory → retrieve → generate → memoize
    """
    
    def __init__(self, context, config: dict, docs_path: str, model_path: str = None, secrets: dict = None):
        """
        Initialize the AgenticAudioModel with configuration and artifacts.
        
        Args:
            config: Model configuration dictionary
            docs_path: Path to documents directory
            model_path: Path to local model file (optional)
            secrets: Dictionary containing secrets (optional)
        """
        self.model_config = config
        self.docs_path = docs_path
        self.model_path = model_path
        self.secrets = secrets
        
        # Initialize components
        self.llm = None
        self.embeddings = None
        self.vectordb = None
        self.retriever = None
        self.chain = None
        self.prompt = None
        self.prompt_str = ""
        self.memory = []
        self.callback_manager = None
        
        # Setup environment and load components
        try:
            self._setup_environment()
            # self._load_embeddings()
            # self._load_vectordb()
            # self._load_model()
            # self._load_prompt()
            # self._load_chain()
            
             # --- Artifacts ---
            index_dir   = Path(context.artifacts["index_dir"])
            config_path = Path(context.artifacts["config_path"])
            memory_dir  = Path(context.artifacts["memory_dir"])
            memory_dir.mkdir(parents=True, exist_ok=True)

            # names (allow env override)
            vecs_name = INDEX_VECS_NPY
            meta_name = INDEX_META_JSON
            
            _set_embeddings(self.docs_path, index_dir, config_path)

            self.vecs = _normalize_vecs(np.load(index_dir / vecs_name).astype(np.float32))
            with open(index_dir / meta_name, "r") as f:
                self.metas = json.load(f)
            with open(config_path, "r") as f:
                cfg = json.load(f)

            self.relevance_threshold = float(cfg.get("relevance_threshold", 0.18))
            self.fetch_k = int(cfg.get("fetch_k", 24))
            self.top_k   = int(cfg.get("top_k", 6))

            # --- CLAP (CPU to avoid OOM) ---
            # keep CLAP on CPU; embed queries quickly and cheaply
            self.clap_processor = ClapProcessor.from_pretrained(cfg["clap_repo"])
            self.clap_model = ClapModel.from_pretrained(cfg["clap_repo"]).eval()
            try:
                self.clap_model.to("cpu")
            except Exception:
                pass

            # --- Memory ---
            # mem_file = os.environ.get("MEMORY_FILENAME", "kv_memory.json")
            self.memory = SimpleKVMemory(memory_dir / MEMORY_FILENAME)

            # --- Qwen Omni (audio agent) ---
            audio_llm_id = os.environ.get("AUDIO_LLM_ID", "Qwen/Qwen2.5-Omni-7B")
            # selector = ModelSelector()
            # local_dir = Path(selector.format_model_path(audio_llm_id))
            # local_dir.mkdir(parents=True, exist_ok=True)

            self.q_processor = Qwen2_5OmniProcessor.from_pretrained(audio_llm_id, trust_remote_code=True)
            self.q_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                audio_llm_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).eval()

            self.audio_llm = _QwenAdapter(self.q_processor, self.q_model)
            self.audio_index = AudioIndex(dim=self.vecs.shape[1])
            self.audio_index.add(self.vecs, self.metas)

            self.graph = build_audio_agentic_graph(
                relevance_threshold = self.relevance_threshold,
                fetch_k = self.fetch_k,
                top_k = self.top_k,
                vecs = self.vecs,
                metas = self.metas,
                audio_index = self.audio_index,
                clap_processor = self.clap_processor,
                clap_model = self.clap_model,
            )
            
            logger.info("AgenticAudioModel initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AgenticAudioModel: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"AgenticAudioModel initialization failed: {str(e)}") from e
    
    def _setup_environment(self) -> None:
        """Configure environment variables based on configuration and secrets."""
        setup_model_environment()
        try:
            # Load secrets into environment if provided
            if self.secrets:
                for key, value in self.secrets.items():
                    os.environ[key] = str(value)
                logger.info("Secrets loaded into environment")
            
            # Configure proxy if specified in config
            if "proxy" in self.model_config and self.model_config["proxy"]:
                logger.info(f"Setting up proxy: {self.model_config['proxy']}")
                os.environ["HTTPS_PROXY"] = self.model_config["proxy"]
                os.environ["HTTP_PROXY"] = self.model_config["proxy"]
            else:
                logger.info("No proxy configuration found")
                    
        except Exception as e:
            logger.error(f"Error setting up environment: {str(e)}")
            # Continue without failing to allow the model to still function
            
    # def _load_model(self) -> None:
    #     """Load the appropriate model based on configuration."""
    #     try:
    #         model_source = self.model_config.get("model_source", "local")
    #         logger.info(f"Loading model with source: {model_source}")
            
    #         from src.utils import initialize_llm, DEFAULT_MODELS
            
    #         # Extract secrets and model path based on configuration
    #         secrets = self.secrets if self.secrets else {}
    #         # Use model_path from notebook if provided, otherwise fall back to default
    #         local_model_path = self.model_path if self.model_path else DEFAULT_MODELS["local"]
    #         logger.info(f"Using local_model_path: {local_model_path}")
            
    #         hf_repo_id = self.model_config.get("hf_repo_id", "")
            
    #         # Use the shared initialize_llm function
    #         self.llm = initialize_llm(
    #             model_source=model_source,
    #             secrets=secrets,
    #             local_model_path=local_model_path,
    #             hf_repo_id=hf_repo_id
    #         )
                
    #         if self.llm is None:
    #             logger.error("Model failed to initialize - llm is None after loading")
    #             raise RuntimeError("Model initialization failed - llm is None")
                
    #         logger.info(f"Model of type {type(self.llm).__name__} loaded successfully")
            
    #     except Exception as e:
    #         logger.error(f"Error loading model: {str(e)}")
    #         raise
    
    # Wrapper used by mlflow
    def _invoke(self, question: str, file_id: str = "global") -> dict:
        return self.graph.invoke({
            "question": question,
            "file_id": file_id,
            "memory": self.memory,
            "audio_llm": self.audio_llm,
            "messages": [],
        })
        
    def predict(self, model_input):
        if isinstance(model_input, pd.DataFrame):
            records = model_input.to_dict(orient="records")
        elif isinstance(model_input, list):
            records = model_input
        else:
            raise ValueError("Pass a list[dict] or pandas DataFrame with 'question' and optional 'file_id'.")

        out = []
        for r in records:
            q  = (r.get("question") or "").strip()
            fid = (r.get("file_id") or "global").strip() or "global"
            try:
                s = self._invoke(q, fid)
                out.append({
                    "question": q,
                    "file_id": fid,
                    "answer": s.get("answer", ""),
                    "evidence": s.get("evidence", []),
                    "from_memory": s.get("from_memory", False),
                })
                
            except Exception as e:
                out.append({
                    "question": q,
                    "file_id": fid,
                    "answer": "",
                    "evidence": [],
                    "from_memory": False,
                    "error": f"{type(e).__name__}: {e}",
                })
        return out
    
    
   
    
   