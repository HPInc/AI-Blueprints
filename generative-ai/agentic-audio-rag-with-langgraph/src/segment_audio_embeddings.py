import numpy as np
import soundfile as sf
import torch
import torchaudio
import subprocess
import faiss
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from src.utils import logger
from langchain_core.documents import Document
from src.generate_test_audio import ensure_ffmpeg_bin, ffmpeg_ok


class AudioIndex:
    def __init__(self, dim: int = 512):
        self.index = faiss.IndexFlatIP(dim)
        self.meta: List[Dict[str, Any]] = []

    def add(self, vecs: np.ndarray, metas: List[Dict[str, Any]]):
        self.index.add(vecs)
        self.meta.extend(metas)

    def search(self, qvec: np.ndarray, k: int = 8) -> List[Dict[str, Any]]:
        D, I = self.index.search(qvec[np.newaxis, :], k)
        out = []
        for idx, score in zip(I[0], D[0]):
            if 0 <= idx < len(self.meta):
                m = dict(self.meta[idx])
                m["score"] = float(score)
                out.append(m)
        return out


FFMPEG_BIN = ensure_ffmpeg_bin()


def _resample_numpy(wav: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return wav.astype(np.float32, copy=False)
    # simple linear interpolation to avoid torchaudio dependency
    n_to = int(round(len(wav) * sr_to / sr_from))
    x = np.linspace(0, 1, num=len(wav), endpoint=False, dtype=np.float64)
    y = np.interp(
        np.linspace(0, 1, num=n_to, endpoint=False), x, wav.astype(np.float64)
    )
    return y.astype(np.float32)


def ensure_wav(
    AUDIO_EXTS: List[str],
    VIDEO_EXTS: List[str],
    src_path: str | Path,
    sr: int = 16000,
    mono: bool = True,
) -> str:
    """
    Ensure a WAV (mono, sr Hz) version exists for any audio/video input.
    Returns the WAV file path.
    """
    src = Path(src_path)
    suffix = src.suffix.lower()
    dst = src.with_suffix(".wav")

    # Fast path: already a WAV; (optionally) enforce mono/sr if you like
    if suffix == ".wav" and dst.exists():
        return str(dst)

    # Prefer ffmpeg for *anything* that's not already a matching WAV
    if ffmpeg_ok():
        exe = (
            os.environ.get("IMAGEIO_FFMPEG_EXE") or shutil.which("ffmpeg") or FFMPEG_BIN
        )
        cmd = [
            exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-ac",
            "1" if mono else "2",
            "-ar",
            str(sr),
            str(dst),
        ]
        import subprocess

        try:
            subprocess.run(cmd, check=True)
            return str(dst)
        except Exception as e:
            # fall through to soundfile path for *audio* formats only
            if suffix in VIDEO_EXTS:
                raise RuntimeError(
                    f"ffmpeg failed to extract audio from video: {src}"
                ) from e

    # Fallback path (no ffmpeg): only for audio inputs libsndfile can read
    if suffix in AUDIO_EXTS:
        data, sr_in = sf.read(str(src))
        if data.ndim == 2:
            data = data.mean(axis=1)
        data = _resample_numpy(data.astype(np.float32, copy=False), sr_in, sr)
        sf.write(str(dst), data, sr)
        return str(dst)

    # If we got here, it's a video file without ffmpeg → cannot proceed
    raise RuntimeError(
        f"Cannot extract audio from video without ffmpeg: {src.name}. "
        "Install ffmpeg or enable imageio-ffmpeg."
    )


def segment_audio(
    wav_path: str, window_s: float = 20.0, hop_s: float = 10.0
) -> List[Tuple[int, int, np.ndarray, int]]:
    """
    Return a list of segments as (start_sample, end_sample, waveform[np.float32 mono], sr).
    """
    audio, sr = sf.read(wav_path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    n = len(audio)
    win = int(window_s * sr)
    hop = int(hop_s * sr)
    if n == 0:
        return []
    segs, i = [], 0
    while i < n:
        j = min(i + win, n)
        segs.append((i, j, audio[i:j], sr))
        if j == n:
            break
        i += hop
    return segs


def _resample_to_48k(wav: np.ndarray, sr: int, target_sr: int = 48000) -> np.ndarray:
    """
    Resample mono waveform to 48k for CLAP. Uses torchaudio if available, else numpy fallback.
    """
    if sr == target_sr:
        return wav.astype(np.float32, copy=False)
    try:
        t = torch.as_tensor(wav, dtype=torch.float32).unsqueeze(0)  # [1, n]
        t48 = torchaudio.functional.resample(t, sr, target_sr)
        return t48.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception:
        # Linear interpolation fallback
        x = np.linspace(0, 1, num=wav.shape[0], dtype=np.float64, endpoint=False)
        y = np.interp(
            np.linspace(
                0, 1, num=int(round(wav.shape[0] * target_sr / sr)), endpoint=False
            ),
            x,
            wav.astype(np.float64, copy=False),
        )
        return y.astype(np.float32)


def _clap_current_device(clap_model):
    try:
        return next(clap_model.parameters()).device
    except Exception:
        return torch.device("cpu")


@torch.no_grad()
def clap_embed_text(clap_processor, clap_model, query: str) -> np.ndarray:
    dev = _clap_current_device(clap_model)
    inp = clap_processor(text=[query], return_tensors="pt")
    inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}
    out = clap_model.get_text_features(**inp)
    vec = out.detach().cpu().numpy()[0]
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec.astype(np.float32)


@torch.no_grad()
def clap_embed_audio(
    clap_processor, clap_model, wav: np.ndarray, sr: int
) -> np.ndarray:
    dev = _clap_current_device(clap_model)
    wav48 = _resample_to_48k(wav, sr, 48000)
    inp = clap_processor(audios=[wav48], sampling_rate=48000, return_tensors="pt")
    inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}
    out = clap_model.get_audio_features(**inp)
    vec = out.detach().cpu().numpy()[0]
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec.astype(np.float32)


def segment_audio_embeddings(
    clap_processor,
    clap_model,
    DATA_PATH: str,
    MEDIA_EXTS: List[str],
    AUDIO_EXTS: List[str],
    VIDEO_EXTS: List[str],
):
    """
    Segment audio files in DATA_PATH, extract embeddings using CLAP, and build an index.
    DATA_PATH: Directory containing media files OR path to a single media file.
    MEDIA_EXTS: List of valid media file extensions.
    AUDIO_EXTS: List of valid audio file extensions.
    VIDEO_EXTS: List of valid video file extensions.
    """

    # Build index over DATA_PATH
    audio_index = AudioIndex(dim=512)  # CLAP audio/text proj dim is 512
    docs_for_ui: List[Document] = []

    data_path = Path(DATA_PATH)

    # Handle both single file and directory
    if data_path.is_file():
        # Single file mode (for API on-demand processing)
        media_paths = [data_path] if data_path.suffix.lower() in MEDIA_EXTS else []
    else:
        # Directory mode (for notebook batch processing)
        media_paths = []
        for p in sorted(data_path.rglob("*")):
            if any(
                part.startswith(".") and part not in {".", ".."} for part in p.parts
            ):
                continue
            if p.is_file() and p.suffix.lower() in MEDIA_EXTS:
                media_paths.append(p)

    for media_path in media_paths:
        wav = ensure_wav(
            AUDIO_EXTS, VIDEO_EXTS, str(media_path)
        )  # OK if this is 16k; we upsample to 48k per segment for CLAP
        segs = segment_audio(wav, window_s=30.0, hop_s=15.0)

        vecs, metas = [], []
        for s0, s1, wav_seg, sr in segs:
            v = clap_embed_audio(clap_processor, clap_model, wav_seg, sr)
            vecs.append(v)
            start_s, end_s = s0 / sr, s1 / sr
            metas.append(
                {
                    "file_path": str(media_path),
                    "file_name": media_path.name,
                    "start_s": float(start_s),
                    "end_s": float(end_s),
                    "wav_path": wav,
                }
            )

        if vecs:
            audio_index.add(np.stack(vecs, axis=0), metas)

        if segs:
            duration_s = segs[-1][1] / segs[-1][3]
            docs_for_ui.append(
                Document(
                    page_content=f"[Audio] {media_path.name} ({duration_s:.1f}s)",
                    metadata={
                        "file_path": str(media_path),
                        "file_name": media_path.name,
                        "media_type": (
                            "audio"
                            if media_path.suffix.lower() in AUDIO_EXTS
                            else "video"
                        ),
                        "segments": [
                            {"start": 0.0, "end": float(duration_s), "text": ""}
                        ],
                        "source": "clap-index",
                    },
                )
            )

    return audio_index, media_paths


def retrieve_audio_segments_from_index(
    audio_index,
    clap_processor,
    clap_model,
    query: str,
    *,
    top_k: int = 6,
    fetch_k: int | None = None,
) -> list[dict]:
    """
    Search the existing index using CLAP embeddings
    """
    if not getattr(audio_index, "meta", None):
        return []
    k = int(fetch_k or top_k)
    qvec = clap_embed_text(clap_processor, clap_model, query)
    return audio_index.search(qvec, k=k)


# --- Reranker: MMR over CLAP embeddings for top-N candidates ---


def _extract_window(
    wav_path: str, start_s: float, end_s: float
) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(wav_path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    i0 = max(0, int(start_s * sr))
    i1 = max(i0, int(end_s * sr))
    return audio[i0:i1].astype(np.float32, copy=False), sr


def rerank_hits_mmr(
    clap_processor,
    clap_model,
    query: str,
    hits: list[dict],
    top_k: int = 6,
    fetch_k: int = 24,
    lam: float = 0.6,
) -> list[dict]:
    """
    Two-stage reranking:
      1) Start from the first `fetch_k` retrievals.
      2) Re-embed each candidate window with CLAP's audio encoder and apply MMR.
         score = λ * sim(query, cand) - (1-λ) * max_j sim(cand, chosen_j)
    Returns the final top_k hits in ranked order with 'score_mmr'.
    """
    if not hits:
        return []

    # Stage-0: take first fetch_k candidates
    cands = hits[: max(fetch_k, top_k)]

    # Query vector in CLAP (text encoder)
    qvec = clap_embed_text(clap_processor, clap_model, query)
    qvec = qvec / (np.linalg.norm(qvec) + 1e-12)

    # Re-embed each candidate on the exact window using CLAP audio encoder
    cand_vecs = []
    for h in cands:
        wav_seg, sr = _extract_window(h["wav_path"], h["start_s"], h["end_s"])
        if wav_seg.size == 0:
            cand_vecs.append(None)
            continue
        v = clap_embed_audio(clap_processor, clap_model, wav_seg, sr)
        v = v / (np.linalg.norm(v) + 1e-12)
        cand_vecs.append(v)

    # Filter out any empty/failed embeddings
    kept = [
        (i, h, v) for i, (h, v) in enumerate(zip(cands, cand_vecs)) if v is not None
    ]
    if not kept:
        return hits[:top_k]  # fallback to original order

    idxs, cands, cand_vecs = zip(*kept)
    cand_vecs = np.stack(cand_vecs, axis=0)

    # MMR selection
    chosen_idx = []
    chosen = []
    avail = set(range(len(cands)))

    while avail and len(chosen) < min(top_k, len(cands)):
        best_i, best_score = None, -1e9
        for i in avail:
            rel = float(np.dot(qvec, cand_vecs[i]))
            div = (
                0.0
                if not chosen_idx
                else max(float(np.dot(cand_vecs[i], cand_vecs[j])) for j in chosen_idx)
            )
            score = lam * rel - (1.0 - lam) * div
            if score > best_score:
                best_i, best_score = i, score
        chosen_idx.append(best_i)
        item = dict(cands[best_i])
        item["score_mmr"] = float(best_score)
        chosen.append(item)
        avail.remove(best_i)

    return chosen


# --- Reranker: MMR over CLAP embeddings for top-N candidates ---
def retrieve_and_rerank(
    audio_index,
    clap_processor,
    clap_model,
    query: str,
    *,
    fetch_k: int = 24,
    top_k: int = 6,
    lam: float = 0.6,
) -> list[dict]:
    # High-recall ANN fetch
    hits = retrieve_audio_segments_from_index(
        audio_index, clap_processor, clap_model, query, top_k=top_k, fetch_k=fetch_k
    )
    if not hits:
        return []
    # Re-embed candidate windows and apply MMR
    return rerank_hits_mmr(
        clap_processor, clap_model, query, hits, top_k=top_k, fetch_k=fetch_k, lam=lam
    )
