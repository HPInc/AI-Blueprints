

from __future__ import annotations

import os
import sys
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

# Import path utilities from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from utils import get_project_root, get_config_dir, get_output_dir, get_default_model_path, get_model_cache_dir



_CONFIG_FILENAMES = {
    "multi": "default_config_multi-gpu.yaml",
    "single": "default_config_one-gpu.yaml",
    "cpu": "default_config-cpu.yaml",
}


def _find_config_dir() -> Path:
    """
    Find the configuration directory containing the required config files.
    
    Returns:
        Path to the directory containing the config files.
    
    Raises:
        FileNotFoundError: If config directory cannot be found.
    """
    # Use the simple config directory function
    config_dir = get_config_dir()
    if config_dir.exists():
        return config_dir
    
    # Fallback to the original search logic if simple approach fails
    required = set(_CONFIG_FILENAMES.values())

    for base in [Path.cwd(), *Path.cwd().parents]:
        if base.is_dir() and required.issubset({p.name for p in base.iterdir()}):
            return base
        cfg_sub = base / "config"
        if cfg_sub.is_dir() and required.issubset({p.name for p in cfg_sub.iterdir()}):
            return cfg_sub

    raise FileNotFoundError(
        f"I did not find a directory with {', '.join(required)} starting from {Path.cwd()}"
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

    # Use the new path utility to get the default model path
    default_model_path = get_default_model_path()
    model_path = cfg.get("model_path", default_model_path)
    
    # Set cache directory for downloaded models
    cache_dir = get_model_cache_dir()
    os.environ["HF_HOME"] = str(cache_dir)

    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="balanced",
        max_memory=max_memory,
        cache_dir=cache_dir,
    )

    inference_times: list[float] = []
    images: list[Image.Image] = []
    
    # Get output directory for saving images
    output_dir = get_output_dir()

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

            # Save image to the output directory
            img_path = output_dir / f"result_{i}.png"
            output_obj.images[0].save(img_path)
            images.append(output_obj.images[0])
            print(f"Saved image to: {img_path}")

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
