#!/usr/bin/env python3
"""
Triton client for enc_dec_CTC (Citrinet) with chunking + automatic padding.
"""

import io
import json
import time
import numpy as np
import requests
import soundfile as sf
import torch
import torchaudio.transforms as T
from pydub import AudioSegment

# === Config ===
TRITON_URL = "http://localhost:8000"
MODEL_NAME = "enc_dec_CTC"
MODEL_INFER_URL = f"{TRITON_URL}/v2/models/{MODEL_NAME}/infer"
MP3_PATH = "audio.mp3" # set our audio file path

CHUNK_DURATION_SEC = 10.0
CHUNK_OVERLAP_SEC = 1.0
SAMPLE_RATE = 16000

# Mel params (Citrinet default)
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 80
F_MIN = 0.0
F_MAX = 8000.0
EPS = 1e-5

STRIDE_TOTAL = 8

def make_mel_transform():
    return T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm=None,
        n_mels=N_MELS,
        mel_scale="htk",
        f_min=F_MIN,
        f_max=F_MAX
    )

def preprocess_waveform(signal_np):
    mel_spec = make_mel_transform()
    with torch.no_grad():
        mel = mel_spec(torch.from_numpy(signal_np.astype(np.float32)))
        mel = torch.log(torch.clamp(mel, min=EPS))
        mel_mean = mel.mean(dim=1, keepdim=True)
        mel_std = mel.std(dim=1, keepdim=True)
        mel = (mel - mel_mean) / (mel_std + EPS)
    return mel

def pad_to_stride(mel, stride_total):
    time_dim = mel.shape[1]
    pad_frames = ((time_dim + stride_total - 1) // stride_total) * stride_total
    if pad_frames > time_dim:
        pad_amount = pad_frames - time_dim
        mel = torch.nn.functional.pad(mel, (0, pad_amount))
    return mel

def chunk_audio(samples, sr, chunk_sec, overlap_sec):
    total_len = samples.shape[0]
    chunk_size = int(chunk_sec * sr)
    step = int((chunk_sec - overlap_sec) * sr)
    for start in range(0, total_len, step):
        end = start + chunk_size
        if start >= total_len:
            break
        if end > total_len:
            end = total_len
        yield start, end, samples[start:end]

def send_to_triton(features_np, lengths_np):
    inputs = [
        {"name": "audio_signal", "shape": list(features_np.shape), "datatype": "FP32", "data": features_np.flatten().tolist()},
        {"name": "length", "shape": list(lengths_np.shape), "datatype": "INT64", "data": lengths_np.tolist()}
    ]
    outputs = [{"name": "logprobs"}]
    body = {"inputs": inputs, "outputs": outputs}
    r = requests.post(MODEL_INFER_URL, headers={"Content-Type": "application/json"}, data=json.dumps(body), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Triton error {r.status_code}: {r.text}")
    return r.json()

def main():
    # Load audio
    audio = AudioSegment.from_file(MP3_PATH)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    signal_np, sr = sf.read(wav_io)
    if signal_np.ndim > 1:
        signal_np = signal_np[:, 0]

    chunks = list(chunk_audio(signal_np, sr, CHUNK_DURATION_SEC, CHUNK_OVERLAP_SEC))
    all_logits = []

    for i, (s, e, chunk_samples) in enumerate(chunks, 1):
        mel = preprocess_waveform(chunk_samples)
        mel = pad_to_stride(mel, STRIDE_TOTAL)

        features_np = mel.unsqueeze(0).numpy().astype(np.float32)  # (1, 80, time)
        lengths_np = np.array([[mel.shape[1]]], dtype=np.int64)

        print(f"Chunk {i}/{len(chunks)}: frames={mel.shape[1]} (padded stride {STRIDE_TOTAL})")
        resp = send_to_triton(features_np, lengths_np)

        logprobs_entry = next(o for o in resp["outputs"] if o["name"] == "logprobs")
        logits = np.array(logprobs_entry["data"]).reshape(logprobs_entry["shape"])
        all_logits.append(logits[0])

        time.sleep(0.05)

    # Concatenate all logits over time
    final_logits = np.concatenate(all_logits, axis=0)
    print(final_logits)
if __name__ == "__main__":
    main()
