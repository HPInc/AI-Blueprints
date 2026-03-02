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

Input schema:
    prompt              (str)   — Text description of the image to generate
    num_inference_steps (int)   — Denoising steps; higher = better quality, slower (default 28)
    guidance_scale      (float) — How strongly the model follows the prompt (default 3.5)
    height              (int)   — Output image height in pixels (default 1024)
    width               (int)   — Output image width in pixels (default 1024)
    seed                (int)   — Fixed seed for reproducibility; -1 = random (default -1)

Output schema:
    answer   (str) — Base64-encoded PNG image (decoded and displayed by Streamlit)
    messages (str) — JSON-serialized request metadata including parameters used
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

    Seed behaviour:
        seed = -1  → random seed (different image every call)
        seed ≥ 0   → deterministic: same prompt + same seed always produces the same image.
                     Useful for iterating on a prompt while keeping composition stable.
    """

    prompt: str = "A beautiful mountain landscape, photorealistic, golden hour lighting"
    num_inference_steps: int = 28  # 20–50 range; higher = better quality, slower
    guidance_scale: float = 3.5  # 1.0–10.0; higher = more prompt-adherent
    height: int = 1024  # Output height in pixels
    width: int = 1024  # Output width in pixels
    seed: int = -1  # -1 = random; ≥0 = fixed deterministic seed


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
            1. Load FLUX.1-dev pipeline (once, cached after first call)
            2. Run text-to-image generation with the requested parameters
            3. Encode the resulting PIL image as base64 PNG
            4. Return the base64 string in the 'answer' column

        Args:
            model_input: DataFrame with required column: prompt.
            params:      Optional dict of generation parameters (MLflow ParamSchema).
                         These are batch-level defaults; any matching column in a
                         DataFrame row overrides the param value for that row.
                         Supported keys (all optional — ImageGenInput provides defaults):
                           num_inference_steps (int)   default 28
                           guidance_scale      (float) default 3.5
                           height              (int)   default 1024
                           width               (int)   default 1024
                           seed                (int)   default -1 (random)

        Returns:
            DataFrame with columns: answer (base64 PNG or error string), messages (str)

        Why use params instead of extra DataFrame columns?
            MLflow's signature validation strips any DataFrame column not listed in the
            input schema before calling predict(). Since only `prompt` is declared as
            required, extra columns would be silently dropped.
            The `params` dict bypasses schema validation and is always forwarded as-is,
            making it the correct MLflow-native way to pass optional/defaulted inputs.
        """
        # params dict supplies batch-level defaults for generation settings.
        # Row-level DataFrame columns (if present) override params for that specific row.
        batch_defaults = dict(params) if params else {}

        results = []

        for _, row in model_input.iterrows():
            try:
                # Merge: batch_defaults < row columns (row wins on conflict).
                # Filter out None/blank-string entries to let ImageGenInput defaults apply.
                row_data = {
                    k: v for k, v in row.items() if v is not None and str(v).strip()
                }
                inp = ImageGenInput(**{**batch_defaults, **row_data})
            except Exception:
                inp = ImageGenInput(
                    prompt=str(row.get("prompt", "")),
                    **{k: v for k, v in batch_defaults.items() if k != "prompt"},
                )

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

                # Step 2: Load the T5-XXL text encoder with 8-bit quantization.
                # T5-XXL is ~9.9 GB in bfloat16 — the largest non-transformer component.
                # Loading it in 8-bit (via bitsandbytes) reduces it to ~5 GB.
                #
                # IMPORTANT: device_map="auto" is required here.
                # bitsandbytes 8-bit modules cannot be moved with .to() after loading,
                # so we let bitsandbytes place the model directly on the GPU via device_map.
                # This also means enable_model_cpu_offload() must NOT be called — it would
                # attempt .to("cpu") on the quantized encoder and raise a RuntimeError.
                from transformers import T5EncoderModel, BitsAndBytesConfig as TF_BnBConfig
                t5_bnb_config = TF_BnBConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bfloat16 for quality
                    bnb_4bit_use_double_quant=True,          # double quantization saves ~0.4 GB extra
                    bnb_4bit_quant_type="nf4",               # NormalFloat4 — best quality for LLM weights
                )
                text_encoder_2 = T5EncoderModel.from_pretrained(
                    self.model_path,
                    subfolder="text_encoder_2",
                    quantization_config=t5_bnb_config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    local_files_only=True,
                )
                logger.info("✅ T5-XXL text encoder loaded (4-bit NF4, ~2.7 GB, placed by device_map)")

                # Step 3: Load the rest of the FLUX pipeline (CLIP text encoder,
                # VAE, scheduler, tokenizers) from the local model directory.
                # The transformer and T5 encoder are replaced with our quantized versions.
                # local_files_only=True prevents any HuggingFace network calls.
                # NOTE: do NOT pass device_map or call .to(device) on the full pipeline —
                # FluxPipeline does not support device_map, and calling .to("cuda") on the
                # whole pipeline would fail because T5 (bitsandbytes 8-bit) cannot be moved.
                logger.info(f"Loading FLUX.1-dev pipeline from: {self.model_path}")
                self._pipeline = FluxPipeline.from_pretrained(
                    self.model_path,
                    transformer=transformer,
                    text_encoder_2=text_encoder_2,
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                )

                # Move each non-quantized component to GPU individually.
                # T5 (text_encoder_2) is already on cuda:0 via device_map="auto" —
                # calling .to() on it would raise a RuntimeError, so skip it.
                # GGUF transformer and vanilla diffusers modules support .to() normally.
                self._pipeline.text_encoder = self._pipeline.text_encoder.to("cuda")
                self._pipeline.transformer  = self._pipeline.transformer.to("cuda")
                self._pipeline.vae          = self._pipeline.vae.to("cuda")

                # Reduce peak activation memory during inference.
                # Static weights (T5 ~2.7 GB + transformer ~6.9 GB + CLIP/VAE ~0.8 GB = ~10.4 GB)
                # are only part of the story — the diffusion loop generates large intermediate
                # tensors (latents, attention maps, VAE decode buffers) for 1024×1024 that can
                # spike an extra 4–6 GB. The options below cut those peaks significantly:
                #
                # vae.enable_slicing()  — decodes the latent image in vertical slices instead
                #   of all at once, reducing VAE peak memory from ~2 GB to ~0.3 GB.
                # vae.enable_tiling()   — tiles VAE encode/decode for very large images;
                #   complements slicing for resolutions above 1024×1024.
                # enable_attention_slicing() — splits attention computation into chunks,
                #   trading a small speed penalty (~5%) for lower attention peak memory.
                self._pipeline.vae.enable_slicing()
                self._pipeline.vae.enable_tiling()
                self._pipeline.enable_attention_slicing()
                logger.info("✅ FLUX.1-dev pipeline loaded — VAE slicing/tiling + attention slicing enabled")

            import torch  # cached after first pipeline load — instant on subsequent calls

            # Build a seeded generator for reproducibility, or None for random output.
            if inp.seed >= 0:
                generator = torch.Generator().manual_seed(inp.seed)
                logger.info(f"Using fixed seed: {inp.seed}")
            else:
                generator = None
                logger.info("Using random seed")

            prompt = (
                inp.prompt
                or "A beautiful mountain landscape, photorealistic, golden hour lighting"
            )
            logger.info(
                f"Generating image — prompt: '{prompt[:60]}...'"
                f" steps={inp.num_inference_steps} guidance={inp.guidance_scale}"
                f" {inp.width}x{inp.height} seed={inp.seed}"
            )

            image = self._pipeline(
                prompt=prompt,
                num_inference_steps=inp.num_inference_steps,
                guidance_scale=inp.guidance_scale,
                height=inp.height,
                width=inp.width,
                generator=generator,
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

            # Free activation memory (intermediate tensors from the diffusion steps).
            # The pipeline itself stays cached for subsequent calls, but releasing
            # unreferenced CUDA tensors here prevents VRAM from creeping up across
            # multiple generations in the same session.
            import gc
            torch.cuda.empty_cache()
            gc.collect()

            logger.info("✅ Image generated successfully")
            return {
                "answer": img_b64,
                "messages": json.dumps(
                    [
                        {
                            "role": "user",
                            "content": prompt,
                            "params": {
                                "num_inference_steps": inp.num_inference_steps,
                                "guidance_scale": inp.guidance_scale,
                                "height": inp.height,
                                "width": inp.width,
                                "seed": inp.seed,
                            },
                        }
                    ]
                ),
            }

        except Exception as e:
            logger.error(f"❌ Image generation error: {e}")
            return {
                "answer": f"❌ Image generation failed: {str(e)}",
                "messages": json.dumps([]),
            }
