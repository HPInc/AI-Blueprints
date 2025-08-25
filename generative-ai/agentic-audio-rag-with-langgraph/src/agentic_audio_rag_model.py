# ─────── Standard Library Imports ───────
from __future__ import annotations # Future-proofing for type annotations
import json  # JSON parsing and serialization
import re # Regular expressions
import logging  # Logging utilities
import multiprocessing  # Multi-process support for concurrency
import os  # Operating system interaction
import time  # Time-related functions
from datetime import datetime  # Date and time manipulation
from pathlib import Path  # Object-oriented filesystem paths
from typing import Any, Dict, List, TypedDict, Annotated  # Static typing support
import numpy as np  # Numerical computing with arrays
import soundfile as sf # Audio file reading and writing
from types import SimpleNamespace # Simple object for attribute access
import operator # Functional programming utilities

# ─────── Third-Party Package Imports ───────
import mlflow  # ML lifecycle platform
import mlflow.pyfunc  # MLflow support for custom Python models
from langchain.docstore.document import Document  # Core document abstraction for LangChain
from langgraph.graph import StateGraph, END  # LangGraph for stateful agent workflows
from pydantic import BaseModel  # Data validation and model parsing
import torch  # PyTorch for deep learning
from transformers import (
    ClapProcessor, ClapModel,
    Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration,
)
from qwen_omni_utils import process_mm_info

# ─────── Local Application-Specific Imports ───────
from src.agentic_workflow import build_audio_agentic_graph  # Custom LangGraph construction logic
from src.simple_kv_memory import SimpleKVMemory, _mem_get, _mem_put  # In-memory key-value store for agent state
from src.model_selection import ModelSelector # Model selection utility
from src.utils import setup_model_environment  # Project-wide configured logger
from src.segment_audio_embeddings import ensure_wav, clap_embed_text

INDEX_VECS_NPY = "audio_vecs.npy"
INDEX_META_JSON = "audio_meta.json"
MEMORY_FILENAME = "kv_memory.json"

 # adapter (same logic as in the notebook, with guards)
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
        import numpy as np, soundfile as sf
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

# def _dot_search(vecs: np.ndarray, metas: List[Dict[str, Any]], qvec: np.ndarray, k: int) -> List[Dict[str, Any]]:
#     q = qvec.astype(np.float32)
#     q = q / (np.linalg.norm(q) + 1e-12)
#     scores = (vecs @ q)
#     idx = np.argsort(-scores)[:k]
#     out = []
#     for i in idx:
#         m = dict(metas[i])
#         m["score"] = float(scores[i])
#         out.append(m)
#     return out

# def _mmr(hits: List[Dict[str, Any]], lam: float = 0.6, top_k: int = 6) -> List[Dict[str, Any]]:
#     # simple score-only MMR placeholder (no cross-sim); preserves input order prioritizing high scores
#     if not hits:
#         return []
#     chosen = []
#     pool = hits.copy()
#     while pool and len(chosen) < top_k:
#         best = max(pool, key=lambda h: h.get("score", 0.0))
#         pool.remove(best)
#         chosen.append(best)
#     # annotate as score_mmr for display
#     for h in chosen:
#         h["score_mmr"] = h.get("score", 0.0)
#     return chosen

class AudioAgenticPyFunc(mlflow.pyfunc.PythonModel):
    """
    Code-based pyfunc model to run the Audio agentic graph.
    Loads artifacts ({index vecs/meta}, config), CLAP (CPU), Qwen Omni, and compiles the graph.
    """

    def load_context(self, context):
        setup_model_environment()

        # --- Artifacts ---
        index_dir   = Path(context.artifacts["index_dir"])
        config_path = Path(context.artifacts["config_path"])
        memory_dir  = Path(context.artifacts["memory_dir"])
        memory_dir.mkdir(parents=True, exist_ok=True)

        # names (allow env override)
        vecs_name = INDEX_VECS_NPY
        meta_name = INDEX_META_JSON

        self.vecs  = _normalize_vecs(np.load(index_dir / vecs_name).astype(np.float32))
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

        self.graph = build_audio_agentic_graph(
            relevance_threshold = self.relevance_threshold,
            fetch_k = self.fetch_k,
            top_k = self.top_k,
            vecs = self.vecs,
            metas = self.metas,
            # clap_processor = self.clap_processor,
            # clap_model = self.clap_model,
        )

       
    # Wrapper used by mlflow
    def _invoke(self, question: str, file_id: str = "global") -> dict:
        return self.graph.invoke({
            "question": question,
            "file_id": file_id,
            "memory": self.memory,
            "audio_llm": self.audio_llm,
            "messages": [],
        })

    def predict(self, context, model_input):
        import pandas as pd
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
