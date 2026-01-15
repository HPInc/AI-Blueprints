"""
Standalone image generation model class containing all business logic.
NO MLflow inheritance - pure domain functionality.
"""

from __future__ import annotations

import base64
import gc
import io
import logging
from pathlib import Path
from typing import Dict, Any, Union, List

import pandas as pd
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline

# Check for xformers availability for memory-efficient attention
try:
    import xformers

    _XFORMERS_AVAILABLE = True
except ImportError:
    _XFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class Model:
    """
    Standalone model class containing all image generation business logic.
    NO MLflow inheritance - pure domain functionality.
    """

    def __init__(
        self,
        model_no_finetuning_path: str,
        model_finetuning_path: str,
        config: Dict[str, Any] = None,
    ):
        """
        Direct dependency injection - no MLflow context.
        Extract all initialization logic from original service.
        """
        self.model_no_finetuning_path = model_no_finetuning_path
        self.model_finetuning_path = model_finetuning_path
        self.config = config or {}

        # Validate model paths and provide helpful information
        self._validate_model_paths()

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus >= 2:
            logger.info("Detected %d GPUs (multi-GPU pipeline)", self.num_gpus)
        elif self.num_gpus == 1:
            logger.info("Detected 1 GPU (single-GPU pipeline)")
        else:
            logger.info("Running on CPU")

        # Initialize as None for lazy loading
        self.current_pipeline = None
        self.current_model = None

        # Clear GPU memory at initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def _validate_model_paths(self):
        """Validate and log information about model paths"""
        base_path = Path(self.model_no_finetuning_path)
        finetuned_path = Path(self.model_finetuning_path)

        logger.info(f"Base model path: {self.model_no_finetuning_path}")
        logger.info(f"Fine-tuned model path: {self.model_finetuning_path}")

        # Check if paths are local directories or HuggingFace model identifiers
        if base_path.exists():
            logger.info("✅ Base model found locally")
        else:
            logger.info("🔄 Base model will be downloaded from HuggingFace Hub")

        if finetuned_path.exists():
            logger.info("✅ Fine-tuned model found locally")
            # Check for common fp16 variant files
            fp16_files = list(finetuned_path.glob("*fp16*"))
            if fp16_files:
                logger.info(
                    f"📁 Found {len(fp16_files)} fp16 variant files in fine-tuned model"
                )
            else:
                logger.warning(
                    "⚠️  No fp16 variant files found in fine-tuned model - will use standard loading"
                )
        else:
            logger.warning("⚠️  Fine-tuned model path does not exist locally")
            logger.warning("    This may cause errors when use_finetuning=true")

    def _load_pipeline(self, use_finetuning: bool):
        """Load the appropriate pipeline based on finetuning preference"""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        need_switch = (
            self.current_pipeline is None
            or (self.current_model == "finetuning" and not use_finetuning)
            or (self.current_model == "no_finetuning" and use_finetuning)
        )
        if not need_switch:
            return

        target = "finetuning" if use_finetuning else "no_finetuning"
        mdl_path = (
            self.model_finetuning_path
            if use_finetuning
            else self.model_no_finetuning_path
        )

        # Memory cleanup before loading new pipeline
        if self.current_pipeline is not None:
            logger.info("Switching pipeline (finetuned = %s)…", use_finetuning)
            del self.current_pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Load pipeline with model-specific handling
        logger.info(
            f"Loading {'fine-tuned' if use_finetuning else 'base'} model from: {mdl_path}"
        )
        self.current_pipeline = self._load_model_with_fallbacks(
            mdl_path, device, use_finetuning
        )

        # Apply memory-efficient setup
        self._setup_pipeline(self.current_pipeline)
        self.current_model = target

        logger.info("Pipeline loaded for %s", target)

    def _load_model_with_fallbacks(
        self, mdl_path: str, device: str, use_finetuning: bool
    ):
        """
        Load SDXL model with targeted fallback strategies.
        Fine-tuned models are more likely to lack fp16 variants than base models.
        """
        model_type = "fine-tuned" if use_finetuning else "base"

        # Strategy 1: Try optimal configuration (fp16 variant + fp16 dtype)
        if torch.cuda.is_available():
            try:
                logger.info(
                    f"Attempting to load {model_type} SDXL model with fp16 variant and dtype"
                )
                return StableDiffusionXLPipeline.from_pretrained(
                    mdl_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    variant="fp16",
                ).to(device)
            except ValueError as e:
                if "variant=fp16" in str(e):
                    logger.warning(
                        f"{model_type.title()} model lacks fp16 variant - this is common for fine-tuned models"
                    )
                else:
                    logger.warning(
                        f"Failed to load {model_type} model with fp16 variant: {e}"
                    )
            except Exception as e:
                logger.warning(
                    f"Unexpected error loading {model_type} model with fp16 variant: {e}"
                )

        # Strategy 2: Try fp16 dtype without variant (most common fallback)
        try:
            logger.info(f"Loading {model_type} SDXL model with fp16 dtype (no variant)")
            return StableDiffusionXLPipeline.from_pretrained(
                mdl_path,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                low_cpu_mem_usage=True,
            ).to(device)
        except Exception as e:
            logger.warning(f"Failed to load {model_type} model with fp16 dtype: {e}")

        # Strategy 3: Try fp32 dtype (compatibility fallback)
        try:
            logger.info(
                f"Loading {model_type} SDXL model with fp32 dtype (compatibility mode)"
            )
            return StableDiffusionXLPipeline.from_pretrained(
                mdl_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
            ).to(device)
        except Exception as e:
            logger.warning(f"Failed to load {model_type} model with fp32 dtype: {e}")

        # Strategy 4: Minimal configuration (last resort)
        try:
            logger.warning(
                f"Loading {model_type} SDXL model with minimal configuration (last resort)"
            )
            return StableDiffusionXLPipeline.from_pretrained(
                mdl_path, low_cpu_mem_usage=True
            ).to(device)
        except Exception as e:
            logger.error(
                f"Failed to load {model_type} model even with minimal configuration: {e}"
            )
            raise RuntimeError(
                f"Unable to load {model_type} model from {mdl_path}. "
                f"Please check if the model path is valid and accessible."
            ) from e

    def _setup_pipeline(self, pipeline):
        """Apply memory-efficient setup to the pipeline"""
        try:
            # Enable memory-efficient attention if xformers is available
            if _XFORMERS_AVAILABLE and hasattr(pipeline, "unet"):
                pipeline.unet.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory-efficient attention")
        except Exception as e:
            logger.warning("Could not enable xformers attention: %s", e)

        try:
            # Enable attention slicing for memory efficiency
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing(slice_size="auto")
                logger.info("Enabled attention slicing")
        except Exception as e:
            logger.warning("Could not enable attention slicing: %s", e)

        try:
            # Enable CPU offloading for large models if needed
            if torch.cuda.is_available() and hasattr(
                pipeline, "enable_sequential_cpu_offload"
            ):
                # Only enable if we have limited GPU memory
                gpu_memory = (
                    torch.cuda.get_device_properties(0).total_memory / 1024**3
                )  # GB
                if gpu_memory < 12:  # For GPUs with less than 12GB
                    pipeline.enable_sequential_cpu_offload()
                    logger.info(
                        "Enabled sequential CPU offloading for memory management"
                    )
        except Exception as e:
            logger.warning("Could not enable CPU offloading: %s", e)

    def predict(
        self, model_input: Union[pd.DataFrame, dict], params: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        Core business logic extracted from original service predict method.
        Remove context parameter - use instance variables instead.
        Must return same pandas.DataFrame structure as original.
        """

        def _first(val):
            return val.iloc[0] if isinstance(val, pd.Series) else val

        prompt = _first(model_input["prompt"])
        use_finetuning = _first(model_input["use_finetuning"])
        # SDXL native resolution is 1024x1024, but support other resolutions
        height = _first(model_input.get("height", 1024))
        width = _first(model_input.get("width", 1024))
        num_images = _first(model_input.get("num_images", 1))
        num_steps = _first(model_input.get("num_inference_steps", 50))

        logger.info("Running inference – '%s'", prompt)
        self._load_pipeline(bool(use_finetuning))

        images64: List[str] = []
        with torch.no_grad():
            for i in range(num_images):
                logger.info("Image %d / %d", i + 1, num_images)
                # Generate image with specified parameters
                img = self.current_pipeline(
                    prompt, height=height, width=width, num_inference_steps=num_steps
                ).images[0]

                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                images64.append(base64.b64encode(buf.read()).decode())
                img.save(f"local_model_result_{i}.png")

                # Clear intermediate GPU memory after each image
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return pd.DataFrame({"output_images": images64})
