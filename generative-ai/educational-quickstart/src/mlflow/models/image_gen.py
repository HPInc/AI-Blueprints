"""
ImageGenModel — Business Logic Layer for Text-to-Image Generation.

Architecture (v2.0.0):
    Focused on one capability: generating images from text prompts.
    Uses FLUX.1-dev with a quantized GGUF transformer via the `diffusers` library.
    Has ZERO MLflow imports — pure Python + PyTorch business logic.

What is FLUX.1-dev?
    FLUX.1-dev is Black Forest Labs' state-of-the-art open-weights text-to-image model.
    It uses a hybrid architecture combining a Multimodal Diffusion Transformer (MMDiT)
    with T5 and CLIP text encoders for superior prompt understanding.

What is a GGUF transformer?
    The GGUF format (from llama.cpp) stores model weights in quantized form.
    For FLUX, `city96/FLUX.1-dev-gguf` provides the transformer block as a single
    quantized .gguf file. Using diffusers >= 0.31, we load this GGUF transformer
    via `FluxTransformer2DModel.from_single_file()` with `GGUFQuantizationConfig`,
    and inject it into a `FluxPipeline` loaded from the rest of the model directory.

    Loading strategy:
        flux1-dev-Q4_K_S.gguf  ← transformer block (GGUF, ~6.9 GB)
        flux1-dev/             ← text encoders + VAE + scheduler (from HF snapshot)

FLUX inference parameters:
    num_inference_steps: 28   (FLUX.1-dev quality sweet-spot; more = better, slower)
    guidance_scale: 3.5       (classifier-free guidance weight; 3-7 typical range)
    height / width: 1024      (FLUX native resolution)

Input schema (focused):
    prompt (str) — Text description of the image to generate

Output schema:
    answer   (str) — Base64-encoded PNG image (decoded and displayed by Streamlit)
    messages (str) — JSON-serialized request metadata
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ─────── Input Schema ────────────────────────────────────────────────────────


class ImageGenInput(BaseModel):
    """
    Input schema for a single image generation request.

    Why base64 for the output image?
        MLflow's pyfunc interface only supports string, number, and array columns.
        Base64 encodes binary image bytes as an ASCII string, making it JSON-safe.
        The Streamlit app decodes this string back into bytes for display.
    """

    prompt: str = "A beautiful mountain landscape, photorealistic, golden hour lighting"


# ─────── Model Class ──────────────────────────────────────────────────────────


class ImageGenModel:
    """
    Text-to-image generator using SDXL-Turbo.

    The pipeline is loaded lazily on the first predict() call to avoid loading
    a multi-GB model before it is actually needed (e.g., during batch registration).

    Key design choice — lazy loading:
        __init__ just stores configuration; the heavy import (torch + diffusers)
        happens inside predict() on the first call. This keeps notebook imports fast.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        docs_path: Optional[str] = None,
        model_path: Optional[str] = None,
        secrets: Optional[Dict] = None,
    ):
        """
        Initialize ImageGenModel.

        Constructor signature is standardized across all v2.0.0 Model classes
        so loader.py can call any of them identically.

        Args:
            config:     Configuration dict from image_gen.yaml
                        (reads model_path for the diffusion pipeline)
            docs_path:  Unused — present for loader.py compatibility
            model_path: Unused for image gen (LLM not needed) — present for compatibility
            secrets:    Optional API keys (not used for local diffusion)
        """
        self.config = config
        self.docs_path = docs_path
        self.secrets = secrets
        # model_path passed explicitly (from loader.py) takes priority;
        # fall back to model_path in config (set by image_gen.yaml or loader.py).
        self.model_path = model_path or config.get(
            "model_path", "/home/jovyan/local/flux1-dev"
        )
        self._pipeline = None  # Populated on first predict() call

    def predict(self, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """
        Generate an image for each row in model_input.

        How it works:
            1. Load SDXL-Turbo pipeline (once, cached after first call)
            2. Run text-to-image generation with 4 denoising steps
            3. Encode the resulting PIL image as base64 PNG
            4. Return the base64 string in the 'answer' column

        Args:
            model_input: DataFrame with column: prompt
            params:      Unused — present for MLflow pyfunc compatibility

        Returns:
            DataFrame with columns: answer (base64 PNG or error), messages (str)
        """
        results = []

        for _, row in model_input.iterrows():
            try:
                inp = ImageGenInput(
                    **{
                        k: str(v)
                        for k, v in row.items()
                        if v is not None and str(v).strip()
                    }
                )
            except Exception:
                inp = ImageGenInput(prompt=str(row.get("prompt", "")))

            results.append(self._generate(inp))

        return pd.DataFrame(results)

    def _generate(self, inp: ImageGenInput) -> dict:
        """Run one inference pass through the FLUX.1-dev GGUF pipeline."""
        import base64
        from io import BytesIO

        gguf_path = os.path.join(self.model_path, "flux1-dev-Q4_K_S.gguf")

        if not os.path.exists(self.model_path):
            return {
                "answer": (
                    f"❌ Image model directory not found at: {self.model_path}\n"
                    "Run project-setup.ipynb Cell 8 to download FLUX.1-dev.\n"
                    "See README.md → Prerequisites."
                ),
                "messages": json.dumps([]),
            }

        if not os.path.exists(gguf_path):
            return {
                "answer": (
                    f"❌ FLUX GGUF transformer not found at: {gguf_path}\n"
                    "Re-run project-setup.ipynb Cell 8 to download flux1-dev-Q4_K_S.gguf."
                ),
                "messages": json.dumps([]),
            }

        transformer_config_path = os.path.join(
            self.model_path, "transformer", "config.json"
        )
        if not os.path.exists(transformer_config_path):
            return {
                "answer": (
                    f"❌ FLUX transformer config not found at: {transformer_config_path}\n"
                    "Re-run project-setup.ipynb step 3c to download transformer/config.json."
                ),
                "messages": json.dumps([]),
            }

        try:
            # Load pipeline once and cache for subsequent inference calls
            if self._pipeline is None:
                import torch
                from diffusers import FluxPipeline, FluxTransformer2DModel

                try:
                    from diffusers import GGUFQuantizationConfig
                except ImportError:
                    from diffusers.utils import GGUFQuantizationConfig

                # Force all HuggingFace Hub calls to use only local files.
                # This prevents from_single_file() fetching the transformer config
                # remotely when all model files are already present on disk.
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"

                # transformer/config.json is downloaded by project-setup.ipynb (step 3c).
                # Pass the directory so from_single_file() uses it directly.
                transformer_config_dir = os.path.join(self.model_path, "transformer")

                logger.info(f"Loading FLUX.1-dev GGUF transformer from: {gguf_path}")

                # Step 1: Load the quantized FLUX transformer from the GGUF file.
                # GGUFQuantizationConfig tells diffusers to use llama.cpp-style
                # quantization when reading weights from the .gguf container.
                quantization_config = GGUFQuantizationConfig(
                    compute_dtype=torch.bfloat16
                )
                transformer = FluxTransformer2DModel.from_single_file(
                    gguf_path,
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                    config=transformer_config_dir,
                )
                logger.info("✅ FLUX GGUF transformer loaded")

                # Step 2: Load the rest of the FLUX pipeline (T5 + CLIP text encoders,
                # VAE, scheduler, tokenizers) from the local model directory.
                # The transformer is replaced with our GGUF version via the kwarg.
                # local_files_only=True prevents any HuggingFace network calls.
                logger.info(f"Loading FLUX.1-dev pipeline from: {self.model_path}")
                self._pipeline = FluxPipeline.from_pretrained(
                    self.model_path,
                    transformer=transformer,
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                )
                # CPU offload moves components to GPU only when needed, then back to CPU.
                # This is necessary for FLUX on systems with < 24 GB VRAM.
                self._pipeline.enable_model_cpu_offload()
                logger.info("✅ FLUX.1-dev pipeline loaded with CPU offload enabled")

            prompt = (
                inp.prompt
                or "A beautiful mountain landscape, photorealistic, golden hour lighting"
            )
            logger.info(f"Generating image for prompt: '{prompt[:60]}...'")

            image = self._pipeline(
                prompt=prompt,
                num_inference_steps=28,  # FLUX.1-dev quality sweet-spot (20–50 range)
                guidance_scale=3.5,  # Classifier-free guidance weight for FLUX
                height=1024,  # FLUX native resolution
                width=1024,
            ).images[0]

            # Embed an invisible watermark to mark AI-generated content (Spec 4.2.3)
            try:
                import numpy as np
                from imwatermark import WatermarkEncoder
                from PIL import Image as PILImage

                wm_encoder = WatermarkEncoder()
                wm_encoder.set_watermark(
                    "bytes", b"AIEQ"
                )  # exactly 4 bytes = 32 bits (rivaGan limit)
                wm_encoder.loadModel()
                img_np = np.array(image.convert("RGB"))
                img_np = wm_encoder.encode(img_np, "rivaGan")
                image = PILImage.fromarray(img_np)
                logger.info("✅ Invisible watermark embedded (rivaGan)")
            except Exception as wm_err:
                logger.warning(
                    "⚠️ Watermark embedding failed (non-critical): %s", wm_err
                )

            # Convert PIL Image → PNG bytes → base64 ASCII string for MLflow transport
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            logger.info("✅ Image generated successfully")
            return {
                "answer": img_b64,
                "messages": json.dumps([{"role": "user", "content": prompt}]),
            }

        except Exception as e:
            logger.error(f"❌ Image generation error: {e}")
            return {
                "answer": f"❌ Image generation failed: {str(e)}",
                "messages": json.dumps([]),
            }
