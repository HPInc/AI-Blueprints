from __future__ import annotations

import logging
import os
import subprocess
import io
import base64
from pathlib import Path
from typing import Union

import mlflow
import torch
import pandas as pd
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from mlflow.types import Schema, ColSpec
from mlflow.models import ModelSignature

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")

class ImageGenerationModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        logging.info("Loading model artefacts…")
        self.model_no_finetuning_path = context.artifacts["model_no_finetuning"]
        self.model_finetuning_path    = context.artifacts["finetuned_model"]

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus >= 2:
            cfg = "config/default_config_multi-gpu.yaml"
            logging.info("Detected %d GPUs → multi-GPU config: %s",
                         self.num_gpus, cfg)
        elif self.num_gpus == 1:
            cfg = "config/default_config_one-gpu.yaml"
            logging.info("Detected 1 GPU → single-GPU config: %s", cfg)
        else:
            cfg = "config/default_config-cpu.yaml"
            logging.info("No GPU detected → CPU config.")
        self.current_pipeline, self.current_model = None, None

    def _load_pipeline(self, use_finetuning: bool):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.current_pipeline is None:
            target = "finetuning" if use_finetuning else "no_finetuning"
        elif    (self.current_model == "finetuning"     and not use_finetuning) \
          or    (self.current_model == "no_finetuning"  and     use_finetuning):
            logging.info("Switching pipeline (finetuned = %s)…", use_finetuning)
            del self.current_pipeline
            torch.cuda.empty_cache()
            target = "finetuning" if use_finetuning else "no_finetuning"
        else:
            return  

        mdl_path = (self.model_finetuning_path if target == "finetuning"
                    else self.model_no_finetuning_path)

        self.current_pipeline = StableDiffusionPipeline.from_pretrained(
            mdl_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device)
        self.current_model = target

    def predict(self, context, X: Union[pd.DataFrame, dict]) -> pd.DataFrame:
        prompt            = X["prompt"].iloc[0]         if isinstance(X, pd.DataFrame) else X["prompt"]
        use_finetuning    = X["use_finetuning"].iloc[0] if isinstance(X, pd.DataFrame) else X["use_finetuning"]
        height            = X.get("height", 512)
        width             = X.get("width",  512)
        num_images        = X.get("num_images", 1)
        num_steps         = X.get("num_inference_steps", 100)

        if isinstance(height, pd.Series):  height  = height.iloc[0]
        if isinstance(width,  pd.Series):  width   = width.iloc[0]
        if isinstance(num_images, pd.Series): num_images = num_images.iloc[0]
        if isinstance(num_steps,  pd.Series): num_steps  = num_steps.iloc[0]

        logging.info("Running inference → \"%s\"", prompt)
        self._load_pipeline(use_finetuning)

        images64: list[str] = []
        with torch.no_grad():
            for i in range(num_images):
                logging.info("Image %d / %d", i + 1, num_images)
                img = self.current_pipeline(
                    prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_steps,
                    guidance_scale=7.5
                ).images[0]

                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                images64.append(base64.b64encode(buf.read()).decode())

                img.save(f"local_model_result_{i}.png")

        return pd.DataFrame({"output_images": images64})

    @classmethod
    def log_model(
        cls,
        finetuned_model_path: str,
        model_no_finetuning_path: str,
        artifact_path: str = "image_generation_model"
    ):
        input_schema  = Schema([
            ColSpec("string",  "prompt"),
            ColSpec("boolean", "use_finetuning"),
            ColSpec("integer", "height"),
            ColSpec("integer", "width"),
            ColSpec("integer", "num_images"),
            ColSpec("integer", "num_inference_steps"),
        ])
        output_schema = Schema([ColSpec("string", "output_images")])
        signature     = ModelSignature(inputs=input_schema, outputs=output_schema)

        core = Path(__file__).resolve().parent.parent
        (core / "__init__.py").touch(exist_ok=True) 

        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=cls(),
            artifacts={
                "finetuned_model":     finetuned_model_path,
                "model_no_finetuning": model_no_finetuning_path,
            },
            signature=signature,
            code_paths=[str(core)], 
            pip_requirements=[
                "torch",
                "diffusers",
                "transformers",
                "accelerate",
                "pillow",
                "pandas",
                "mlflow",
            ],
        )
        logging.info("✅ Model logged to MLflow at «%s»", artifact_path)

def setup_accelerate():
    subprocess.run(["pip", "install", "accelerate"], check=True)
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        cfg = "config/default_config_multi-gpu.yaml"
    elif num_gpus == 1:
        cfg = "config/default_config_one-gpu.yaml"
    else:
        cfg = "config/default_config-cpu.yaml"
    os.environ["ACCELERATE_CONFIG_FILE"] = cfg
    logging.info("Using accelerate cfg: %s", cfg)

def deploy_model():
    setup_accelerate()

    mlflow.set_tracking_uri('/phoenix/mlflow')
    mlflow.set_experiment("ImageGeneration")
    finetuned = "./dreambooth"
    base      = "../../../local/stable-diffusion-2-1"

    with mlflow.start_run(run_name="image_generation_service") as run:
        mlflow.log_artifact(os.environ["ACCELERATE_CONFIG_FILE"],
                            artifact_path="accelerate_config")
        ImageGenerationModel.log_model(
            finetuned_model_path=finetuned,
            model_no_finetuning_path=base,
        )
        model_uri = f"runs:/{run.info.run_id}/image_generation_model"
        mlflow.register_model(model_uri=model_uri,
                              name="ImageGenerationService")
        logging.info("🏷️ Registered «ImageGenerationService» (run %s)",
                     run.info.run_id)

 