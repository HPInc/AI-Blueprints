# ─────── Standard Library Imports ───────
import gc  # Python's garbage collector — frees RAM/VRAM from unused objects
import logging  # Logging system
import multiprocessing  # Used to count CPU cores for optimal thread count
import os  # File path utilities
from pathlib import Path  # Object-oriented file paths (easier than string manipulation)
from typing import Optional, Dict, Any  # Type hints

# Set up module-level logger
logger = logging.getLogger(__name__)

# ─────── Constants ───────────────────────────────────────────────────────────
# Base directory where AI Studio stores downloaded models.
# All model paths in this project use this as the root.
DATAFABRIC_BASE = "/home/jovyan/datafabric"


def verify_model_exists(model_path: str) -> bool:
    """
    Check whether a model file or directory exists at the given path.

    Why call this before loading?
        Loading a model takes 30–120 seconds. Calling this function first gives
        an immediate, clear error if the path is wrong — instead of a confusing
        traceback after a long wait.

    Args:
        model_path: Full path to the model file or directory

    Returns:
        True if the path exists, False otherwise

    Example:
        if not verify_model_exists("/home/jovyan/datafabric/my-model/model.gguf"):
            print("Download the model first! See README.md Step 2.")
    """
    exists = os.path.exists(model_path)
    if exists:
        logger.info(f"✅ Model found: {model_path}")
    else:
        logger.error(
            f"❌ Model not found: {model_path}\n"
            "   Please follow README.md Step 2 to download the model into datafabric."
        )
    return exists


def load_llm(model_path: str, n_ctx: int = 8192, **kwargs):
    """
    Load a quantized LLaMA model using LlamaCpp with GPU-optimized settings.

    What is LlamaCpp?
        LlamaCpp is a highly optimized C++ inference engine for running LLaMA-family
        models locally. It supports quantization (running models in 4-bit or 8-bit
        precision) which dramatically reduces VRAM requirements while maintaining
        good quality.

    What is quantization?
        Training AI models uses 32-bit floating point numbers (FP32).
        Quantization compresses the model weights to 4-bit or 8-bit integers,
        reducing model size by 4–8× with only a small accuracy penalty.
        A "Q8" model (like Meta-Llama-3.1-8B-Q8) uses 8-bit quantization.

    Blackwell-optimized settings used here:
        - n_gpu_layers=-1   → offload ALL layers to VRAM (maximum GPU acceleration)
        - n_batch=512       → process 512 tokens at a time (balances speed vs. memory)
        - f16_kv=True       → use 16-bit floats for keys/values (halves KV-cache memory)
        - use_mmap=False    → don't use memory-mapped files (more reliable on some systems)
        - temperature=0.0   → deterministic output (0 = always pick the most likely token)
        - seed=42           → reproducible results

    Args:
        model_path: Full path to the .gguf model file
        n_ctx: Context window size — how many tokens the model can "remember" at once
               (8192 ≈ ~6,000 words). Larger = more memory usage.
        **kwargs: Additional keyword arguments passed to LlamaCpp

    Returns:
        A LlamaCpp model instance ready for inference

    Learn more:
        https://python.langchain.com/docs/integrations/llms/llamacpp/
        https://github.com/ggerganov/llama.cpp
    """
    try:
        from langchain_community.llms import LlamaCpp  # LangChain wrapper for llama.cpp
    except ImportError:
        raise ImportError("LlamaCpp not found. Run: %pip install langchain-community")

    if not verify_model_exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            "Please download the model first. See README.md Step 2."
        )

    # Get CPU count for optimal threading (use all available cores)
    cpu_count = multiprocessing.cpu_count()
    max_tokens = n_ctx // 8  # Reserve ⅛ of the context for the response

    logger.info(f"Loading LLM from: {model_path}")
    logger.info(f"Context window: {n_ctx} tokens | Max tokens: {max_tokens}")

    # Merge default settings with any user overrides from kwargs
    llm_kwargs = {
        "model_path": model_path,
        "n_gpu_layers": -1,  # -1 = offload ALL layers to GPU
        "n_batch": 512,  # Tokens processed per batch
        "n_ctx": n_ctx,  # Total context window
        "max_tokens": max_tokens,  # Max tokens to generate per call
        "f16_kv": True,  # Use float16 for key-value cache (saves VRAM)
        "use_mmap": False,  # Disable memory-mapped file reading
        "low_vram": False,  # Don't use low-VRAM mode (we offload fully to GPU)
        "temperature": 0.0,  # 0 = deterministic (greedy) decoding
        "repeat_penalty": 1.0,  # Penalize repeated tokens (1.0 = no penalty)
        "streaming": False,  # Return complete response, not token-by-token
        "seed": 42,  # Fixed seed = reproducible outputs
        "num_threads": cpu_count,  # Use all CPU cores for non-GPU operations
        "verbose": False,  # Suppress llama.cpp progress logs
    }
    llm_kwargs.update(kwargs)  # Apply any user-provided overrides

    llm = LlamaCpp(**llm_kwargs)
    logger.info("✅ LLM loaded successfully")
    return llm


def load_diffusion_pipeline(model_path: str, **kwargs):
    """
    Load a text-to-image diffusion model pipeline from the datafabric.

    What is a diffusion model?
        Diffusion models generate images by starting with random noise and
        progressively "denoising" it into a coherent image guided by your text prompt.
        SDXL-Turbo is a fast version that produces good results in just 1–4 steps.

    What is float16?
        Using torch.float16 (16-bit precision instead of 32-bit) halves the VRAM
        required to load the model while having minimal impact on output quality.
        This is a standard optimization for running large image models.

    Args:
        model_path: Full path to the model directory (containing model_index.json)
        **kwargs: Additional arguments passed to AutoPipelineForText2Image

    Returns:
        A diffusers pipeline ready for image generation

    Learn more:
        https://huggingface.co/docs/diffusers/
        https://huggingface.co/stabilityai/sdxl-turbo
    """
    try:
        import torch
        from diffusers import AutoPipelineForText2Image  # Universal pipeline factory
    except ImportError:
        raise ImportError(
            "diffusers or torch not found. Run: %pip install diffusers torch"
        )

    if not verify_model_exists(model_path):
        raise FileNotFoundError(
            f"Diffusion model not found at: {model_path}\n"
            "Please download the model first. See README.md Step 2."
        )

    logger.info(f"Loading diffusion pipeline from: {model_path}")

    # Build kwargs — allow user overrides
    pipeline_kwargs = {
        "torch_dtype": torch.float16,  # Use 16-bit precision to save VRAM
    }
    # Only add "variant" if using fp16 variant files (not all models have them)
    if os.path.exists(
        os.path.join(model_path, "unet", "diffusion_pytorch_model.fp16.safetensors")
    ):
        pipeline_kwargs["variant"] = "fp16"

    pipeline_kwargs.update(kwargs)  # Apply user overrides

    pipe = AutoPipelineForText2Image.from_pretrained(model_path, **pipeline_kwargs)
    pipe = pipe.to("cuda")  # Move all model weights to the GPU

    logger.info("✅ Diffusion pipeline loaded and moved to GPU")
    return pipe


def load_whisper_model(model_path: str, **kwargs):
    """
    Load an OpenAI Whisper model for speech-to-text transcription.

    What is Whisper?
        Whisper is an automatic speech recognition (ASR) model trained by OpenAI
        on 680,000 hours of diverse audio. It can transcribe speech in 99 languages
        and is highly robust to accents, background noise, and technical vocabulary.

        whisper-large-v3 is the most capable variant (1.5B parameters).

    Args:
        model_path: Path to the local Whisper model directory OR a model size string
                    (e.g., "large-v3") for downloading from the internet
        **kwargs: Additional arguments passed to the Whisper loader

    Returns:
        A Whisper model instance

    Learn more:
        https://github.com/openai/whisper
        https://huggingface.co/openai/whisper-large-v3
    """
    try:
        import whisper  # openai-whisper package
    except ImportError:
        raise ImportError("openai-whisper not found. Run: %pip install openai-whisper")

    logger.info(f"Loading Whisper model from: {model_path}")

    if os.path.isdir(model_path):
        # Load from a local directory (downloaded via AI Studio Models tab)
        # Try to find the .pt file inside the directory
        pt_files = list(Path(model_path).glob("*.pt"))
        if pt_files:
            model = whisper.load_model(str(pt_files[0]), **kwargs)
        else:
            logger.warning(
                "⚠️ No .pt file found in model_path directory — trying as model name"
            )
            # Fall back to treating model_path as a size identifier
            model_name = os.path.basename(model_path.rstrip("/"))
            model = whisper.load_model(model_name, **kwargs)
    else:
        # model_path is a size string like "large-v3" or "base"
        model = whisper.load_model(model_path, **kwargs)

    logger.info("✅ Whisper model loaded successfully")
    return model


def get_quantization_config(model_size_b: float) -> Dict[str, Any]:
    """
    Return a BitsAndBytes quantization configuration appropriate for the model size.

    What is BitsAndBytes?
        BitsAndBytes is a library that enables loading Hugging Face Transformer models
        in 4-bit or 8-bit precision, dramatically reducing VRAM requirements.

    Quantization strategy by model size:
        < 13B parameters  → 4-bit NF4   (smallest VRAM footprint)
        13B–70B           → 8-bit       (balance of quality and VRAM)
        ≥ 70B             → 4-bit GPTQ  (for very large models, often used with vLLM)

    Args:
        model_size_b: Model size in billions of parameters (e.g., 7.0, 13.0, 70.0)

    Returns:
        A dict with quantization configuration kwargs for from_pretrained()

    Learn more:
        https://huggingface.co/docs/transformers/quantization/bitsandbytes
        https://github.com/TimDettmers/bitsandbytes
    """
    try:
        import torch
        from transformers import BitsAndBytesConfig  # Quantization config class
    except ImportError:
        logger.warning("⚠️ BitsAndBytes not available — returning empty config")
        return {}

    if model_size_b < 13.0:
        # 4-bit NF4 quantization — best for smaller models under 13B
        # NF4 = "NormalFloat 4-bit", designed specifically for neural network weights
        logger.info(f"Using 4-bit NF4 quantization for {model_size_b}B model")
        config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit loading
            bnb_4bit_quant_type="nf4",  # NF4 quantization type
            bnb_4bit_compute_dtype=torch.float16,  # Do computation in float16
            bnb_4bit_use_double_quant=True,  # Double quantize for extra compression
        )
    elif model_size_b < 70.0:
        # 8-bit quantization — good balance for medium-large models (13B–70B)
        logger.info(f"Using 8-bit quantization for {model_size_b}B model")
        config = BitsAndBytesConfig(
            load_in_8bit=True,  # Enable 8-bit loading
        )
    else:
        # For 70B+ models, 4-bit GPTQ with vLLM is typically recommended
        # Return basic 4-bit config as a fallback
        logger.info(
            f"Using 4-bit NF4 for {model_size_b}B+ model (recommend vLLM for production)"
        )
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    return {"quantization_config": config}


def safe_cuda_cleanup() -> None:
    """
    Safely release GPU memory caches and run Python garbage collection.

    When to call this:
        - After loading a large model (to free any temporary allocations)
        - After an Out-of-Memory (OOM) error (before retrying)
        - Between running multiple models in the same session

    What it does:
        1. torch.cuda.empty_cache()  — tells PyTorch to release its internal memory
           cache back to the GPU driver (the driver can then give it to another process)
        2. gc.collect()              — runs Python's garbage collector, freeing Python
           objects that are no longer referenced

    Important note:
        This frees CACHED memory, not memory actively used by loaded models.
        To fully free a model's memory, you need to delete the model variable first:
            del my_model
            safe_cuda_cleanup()

    Learn more:
        https://pytorch.org/docs/stable/notes/cuda.html#memory-management
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Release PyTorch's VRAM cache
    except ImportError:
        pass  # No PyTorch available — nothing to clean up

    gc.collect()  # Run Python's garbage collector to free CPU RAM too
    logger.info("🧹 CUDA cache cleared and garbage collection complete")
