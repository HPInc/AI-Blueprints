

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from PIL import Image



_CONFIG_FILENAMES = {
    "multi": "default_config_multi-gpu.yaml",
    "single": "default_config_one-gpu.yaml",
    "cpu": "default_config-cpu.yaml",
}


def _find_config_dir() -> Path:
    required = set(_CONFIG_FILENAMES.values())
    for base in [Path.cwd(), *Path.cwd().parents]:
        if required.issubset({p.name for p in base.iterdir()}):
            return base
        cfg = base / "config"
        if cfg.is_dir() and required.issubset({p.name for p in cfg.iterdir()}):
            return cfg
    raise FileNotFoundError(
        f"I did not find a directory with {', '.join(required)} starting from {Path.cwd()}"
    )

class StableDiffusionPipelineOutput:
    def __init__(self, images: Union[List[Image.Image], np.ndarray], nsfw_content_detected: List[bool]):
        self.images = images
        self.nsfw_content_detected = nsfw_content_detected


def load_config_dreambooth(config_dir: str | Path | None = None) -> dict:
    base = (
        Path(config_dir).expanduser() if config_dir else
        Path(os.getenv("CONFIG_DIR", "")).expanduser() if os.getenv("CONFIG_DIR") else
        _find_config_dir()
    )
    n_gpu = torch.cuda.device_count()
    key = "multi" if n_gpu >= 2 else "single" if n_gpu == 1 else "cpu"
    cfg_path = base / _CONFIG_FILENAMES[key]
    print(f"Detected {n_gpu} GPU(s); loading {cfg_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def get_max_memory_per_gpu(reserve_gb: int = 2) -> dict[int, str]:
    mem = {}
    for idx in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
        mem[idx] = f"{int(total - reserve_gb)}GB"
        print(f"GPU {idx}: reserving {mem[idx]} of VRAM.")
    return mem


def display_generated_images(images: List[Image.Image]):
    if not images:
        print("No images to display.")
        return
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(images, 1):
        plt.subplot(1, len(images), i)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Image {i}")
    plt.tight_layout(); plt.show()


def run_inference_dreambooth(
    prompt: str | None = None,
    *,
    height: int = 512,
    width: int = 512,
    num_images: int = 5,
    num_inference_steps: int = 50,
    output: bool = True,
    model_path: str | Path = "./dreambooth",
    config_dir: str | Path | None = None,
):
    """Generates *num_images* using DreamBooth template.."""
    _ = load_config_dreambooth(config_dir) 

    default_prompt = (
        "A sleek, modern laptop open on a sandy beach, positioned in front of a vibrant blue ocean. "
        "Soft shadows, seashells nearby, palm trees in the background, bright tropical vibes."
    )
    prompt = prompt or default_prompt

    accelerator = Accelerator(mixed_precision="fp16", cpu=False)
    max_memory = get_max_memory_per_gpu()

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="balanced",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )

    times, images = [], []
    if accelerator.process_index == 0:
        for i in range(num_images):
            t0 = time.time()
            res = pipe(prompt, height=height, width=width, num_inference_steps=num_inference_steps)
            times.append(time.time() - t0)
            res.images[0].save(f"local_model_result_{i}.png")
            images.append(res.images[0])
            print(f"[{i+1}/{num_images}] {times[-1]:6.2f} s")

        arr = np.array(times)
        print(f"\nAvg {arr.mean():.2f}s | Med {np.median(arr):.2f}s | Min {arr.min():.2f}s | Max {arr.max():.2f}s")
        if output:
            display_generated_images(images)

    accelerator.end_training()


if __name__ == "__main__":
    run_inference_dreambooth(num_images=1)
