

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
from diffusers import DiffusionPipeline
from PIL import Image



_CONFIG_FILENAMES = {
    "multi": "default_config_multi-gpu.yaml",
    "single": "default_config_one-gpu.yaml",
    "cpu": "default_config-cpu.yaml",
}


def _find_config_dir() -> Path:
  
    required = set(_CONFIG_FILENAMES.values())

    for base in [Path.cwd(), *Path.cwd().parents]:
        if base.is_dir() and required.issubset({p.name for p in base.iterdir()}):
            return base
        cfg_sub = base / "config"
        if cfg_sub.is_dir() and required.issubset({p.name for p in cfg_sub.iterdir()}):
            return cfg_sub

    raise FileNotFoundError(
        f"I did not find a directory with {', '.join(required)}  starting from {Path.cwd()}"
    )


def load_config(config_dir: str | Path | None = None) -> dict:
   

    if config_dir:
        base = Path(config_dir).expanduser()
    elif os.getenv("CONFIG_DIR"):
        base = Path(os.getenv("CONFIG_DIR")).expanduser()
    else:
        base = _find_config_dir()

    num_gpus = torch.cuda.device_count()
    key = "multi" if num_gpus >= 2 else "single" if num_gpus == 1 else "cpu"
    cfg_path = base / _CONFIG_FILENAMES[key]

    print(f"Detected {num_gpus} GPU(s); loading {cfg_path}")

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)




class StableDiffusionPipelineOutput:
    def __init__(self, images: Union[List[Image.Image], np.ndarray], nsfw_content_detected: List[bool]):
        self.images = images
        self.nsfw_content_detected = nsfw_content_detected


def _get_max_memory_per_gpu() -> dict[int, str]:
    """Reserves ~2 GB below each GPU's total VRAM to avoid OOM"""
    max_mem: dict[int, str] = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        max_mem[i] = f"{int(props.total_memory / 1024**3 - 2)}GB"
        print(f"GPU {i}: reserving {max_mem[i]} ofix VRAM.")
    return max_mem


def _display_images(images: List[Image.Image]) -> None:
    if not images:
        print("No images to display.")
        return

    plt.figure(figsize=(15, 5))
    for idx, img in enumerate(images, 1):
        plt.subplot(1, len(images), idx)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Image {idx}")
    plt.tight_layout()
    plt.show()


def run_inference(
    prompt: str | None = None,
    *,
    height: int = 512,
    width: int = 512,
    num_images: int = 5,
    num_inference_steps: int = 50,
    output: bool = True,
    config_dir: str | Path | None = None,
) -> None:

    cfg = load_config(config_dir)

    default_prompt = (
        "A sleek, modern laptop open on a sandy beach, positioned in front of a vibrant blue ocean. "
        "Soft shadows, seashells nearby, palm trees in the background, bright tropical vibes."
    )
    prompt = prompt or default_prompt

    accelerator = Accelerator(mixed_precision="fp16", cpu=False)
    max_memory = _get_max_memory_per_gpu()

    model_path = cfg.get("model_path", "../../../local/stable-diffusion-2-1/")

    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="balanced",
        max_memory=max_memory,
    )

    inference_times: list[float] = []
    images: list[Image.Image] = []

    if accelerator.is_main_process:
        for i in range(num_images):
            t0 = time.time()
            result = pipe(
                prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
            )
            t1 = time.time()

            output_obj = StableDiffusionPipelineOutput(
                images=result.images, nsfw_content_detected=[False] * len(result.images)
            )

            inf_time = t1 - t0
            inference_times.append(inf_time)
            print(f"[{i+1}/{num_images}] {inf_time:6.2f} s")

            img_path = f"result_{i}.png"
            output_obj.images[0].save(img_path)
            images.append(output_obj.images[0])

        arr = np.array(inference_times)
        print(
            f"\nAverage: {arr.mean():.2f}s | Median: {np.median(arr):.2f}s | "
            f"Min: {arr.min():.2f}s | Max: {arr.max():.2f}s"
        )

        if output:
            _display_images(images)

    accelerator.end_training()




if __name__ == "__main__":
    run_inference(num_images=3)
