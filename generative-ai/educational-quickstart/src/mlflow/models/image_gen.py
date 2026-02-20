"""
ImageGenModel — Business Logic Layer for Text-to-Image Generation.

Architecture (v2.0.0):
    Focused on one capability: generating images from text prompts.
    Uses the SDXL-Turbo diffusion model via the `diffusers` library.
    Has ZERO MLflow imports — pure Python + PyTorch business logic.

What is SDXL-Turbo?
    SDXL-Turbo is a distilled version of Stable Diffusion XL that generates
    high-quality images in just 1-4 denoising steps (vs. 20-50 for standard SD).
    This makes it ideal for interactive demos and educational environments.

What is a diffusion model?
    A diffusion model learns to generate images by reversing a noise-adding process.
    During training, it learns to "denoise" random noise into coherent images.
    At inference time, it starts with random noise and iteratively denoises it
    guided by your text prompt into a realistic image.

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
                        (reads image_model_path for the diffusion pipeline)
            docs_path:  Unused — present for loader.py compatibility
            model_path: Unused for image gen (LLM not needed) — present for compatibility
            secrets:    Optional API keys (not used for local diffusion)
        """
        self.config = config
        self.docs_path = docs_path
        self.model_path = model_path
        self.secrets = secrets

        # Resolved at init time; pipeline itself loaded lazily in predict()
        self.image_model_path = config.get(
            "image_model_path",
            "/home/jovyan/datafabric/sdxl-turbo",
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
        """Run one inference pass through the SDXL-Turbo pipeline."""
        import base64
        from io import BytesIO

        if not os.path.exists(self.image_model_path):
            return {
                "answer": (
                    f"❌ Image model not found at: {self.image_model_path}\n"
                    "Download 'sdxl-turbo' (stabilityai/sdxl-turbo) into datafabric.\n"
                    "See README.md → Prerequisites."
                ),
                "messages": json.dumps([]),
            }

        try:
            # Load pipeline once and reuse for subsequent calls
            if self._pipeline is None:
                import torch
                from diffusers import AutoPipelineForText2Image

                logger.info(
                    f"Loading SDXL-Turbo pipeline from: {self.image_model_path}"
                )
                self._pipeline = AutoPipelineForText2Image.from_pretrained(
                    self.image_model_path,
                    torch_dtype=torch.bfloat16,  # BF16 avoids cuBLAS FP16 alignment issues on CUDA 12.x
                ).to("cuda")
                logger.info("✅ Diffusion pipeline loaded")

            prompt = inp.prompt or "A beautiful landscape"
            logger.info(f"Generating image for prompt: '{prompt[:60]}...'")

            image = self._pipeline(
                prompt=prompt,
                num_inference_steps=4,  # SDXL-Turbo quality degrades above ~4 steps
                guidance_scale=0.0,  # Turbo style: no classifier-free guidance
            ).images[0]

            # Convert PIL Image → PNG bytes → base64 ASCII string
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
