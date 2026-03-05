"""
Shared post-inference VRAM cleanup utilities for EQ model classes.

Why this module exists
----------------------
When four MLflow model servers (chatbot, document, image_gen, voice) run
simultaneously on the same GPU, each server's PyTorch CUDA caching allocator
retains freed memory blocks instead of returning them to the CUDA driver.
Over successive inferences this causes:

  1. **VRAM fragmentation** — PyTorch cannot assemble a large contiguous region
     for FLUX.1-dev latent tensors (1024×1024 × 28 steps), so the allocator
     attempts a fallback that silently corrupts diffusion state → irrelevant images.

  2. **llama.cpp KV-cache starvation** — when the driver cannot satisfy a large
     KV-cache allocation because PyTorch's cached blocks occupy the needed VRAM,
     llama.cpp falls back to a truncated or partially-offloaded KV-cache.  The
     attention mechanism degrades and the model outputs degenerate tokens
     (the classic "!!!..." repetition pattern).

This module provides two lightweight helpers that every Model class calls at the
boundary of each inference request:

  ``cuda_cleanup(label)``           — flushes post-inference garbage
  ``log_vram_pre_inference(label)`` — debug-level VRAM snapshot before inference

These are intentionally NOT ``release_model_vram()`` from ``src.utils``.
That function destroys model weights entirely (used between notebook sections).
These helpers only release *intermediate tensors* that accumulated during one
inference pass; the model stays warm and resident in VRAM.

Usage (inside any Model class method)
--------------------------------------
    from src.mlflow.models._cleanup import cuda_cleanup, log_vram_pre_inference

    log_vram_pre_inference("chatbot")
    answer = chain.invoke(...)
    cuda_cleanup("chatbot")
"""

import gc
import logging

logger = logging.getLogger(__name__)


def cuda_cleanup(label: str = "") -> None:
    """
    Release intermediate CUDA tensors accumulated during one inference call.

    Call this **after** the inference result has been collected and **before**
    returning from the model method.  The model's own weights are not affected.

    Sequence of operations:
        1. ``gc.collect()``             — Python cycle-collector breaks reference
                                          cycles that keep CUDA tensors alive past
                                          their logical lifetime.
        2. ``torch.cuda.synchronize()`` — blocks until every pending GPU kernel
                                          completes so no in-flight op still holds
                                          a reference to a tensor about to be freed.
        3. ``torch.cuda.empty_cache()`` — returns all freed blocks from PyTorch's
                                          CUDA caching allocator back to the CUDA
                                          driver, immediately making them available
                                          to other processes (e.g., FLUX needing a
                                          4-6 GB contiguous latent buffer).
        4. ``gc.collect()``             — second pass because step 1's ``__del__``
                                          chains may themselves allocate short-lived
                                          Python objects that are now collectable.

    Args:
        label: Human-readable tag shown in debug logs (e.g., ``"chatbot"``,
               ``"image_gen-preflight"``). Empty string produces no tag prefix.
    """
    try:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

        if logger.isEnabledFor(logging.DEBUG) and torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            used_gb = (total_b - free_b) / 1024**3
            total_gb = total_b / 1024**3
            logger.debug(
                "🧹 VRAM after %sinference: %.1f / %.1f GB (%.0f%% used)",
                f"{label} " if label else "",
                used_gb,
                total_gb,
                used_gb / total_gb * 100,
            )
    except Exception:
        # Never let cleanup failures propagate and break inference.
        pass


def log_vram_pre_inference(label: str = "") -> None:
    """
    Log available VRAM immediately before an inference call (debug level).

    Captures a snapshot of free vs. used VRAM *before* any inference allocation
    begins.  Comparing this snapshot against the post-inference log from
    ``cuda_cleanup()`` reveals whether each request cleaned up fully or left
    a growing residue — the telltale sign of a memory leak.

    Args:
        label: Human-readable tag (e.g., ``"image_gen"``, ``"voice-llm"``).
    """
    try:
        import torch

        if torch.cuda.is_available() and logger.isEnabledFor(logging.DEBUG):
            free_b, total_b = torch.cuda.mem_get_info()
            free_gb = free_b / 1024**3
            used_gb = (total_b - free_b) / 1024**3
            total_gb = total_b / 1024**3
            logger.debug(
                "📊 VRAM before %sinference: %.1f GB free / %.1f GB used / %.1f GB total",
                f"{label} " if label else "",
                free_gb,
                used_gb,
                total_gb,
            )
    except Exception:
        pass
