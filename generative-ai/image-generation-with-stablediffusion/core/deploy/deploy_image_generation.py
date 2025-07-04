

from __future__ import annotations

import base64, io, logging, os, shutil, subprocess, sys, time
from pathlib import Path
from typing import Union

import mlflow, torch, pandas as pd, numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from mlflow.types import Schema, ColSpec
from mlflow.models import ModelSignature

# Import path utilities from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from utils import get_project_root, get_config_dir, get_output_dir

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")



_CONFIG_FILENAMES = {
    "multi": "default_config_multi-gpu.yaml",
    "single": "default_config_one-gpu.yaml",
    "cpu": "default_config-cpu.yaml",
}

def _find_config_dir() -> Path:
    """Find the config directory using simple relative path resolution"""
    # Try the simple approach first
    config_dir = get_config_dir()
    if config_dir.exists():
        return config_dir
    
    # Fallback to searching
    required = set(_CONFIG_FILENAMES.values())
    for base in [Path.cwd(), *Path.cwd().parents]:
        if required.issubset({p.name for p in base.iterdir()}):
            return base
        cfg = base / "config"
        if cfg.is_dir() and required.issubset({p.name for p in cfg.iterdir()}):
            return cfg
    raise FileNotFoundError(
        f"I did not find a directory with{', '.join(required)} starting from{Path.cwd()}"
    )



class ImageGenerationModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        logging.info("Loading model artefacts…")
        self.model_no_finetuning_path = context.artifacts["model_no_finetuning"]
        self.model_finetuning_path    = context.artifacts["finetuned_model"]

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus >= 2:
            logging.info("Detected %d GPUs (multi-GPU pipeline)", self.num_gpus)
        elif self.num_gpus == 1:
            logging.info("Detected 1 GPU (single-GPU pipeline)")
        else:
            logging.info("Running on CPU")
        self.current_pipeline, self.current_model = None, None

    def _load_pipeline(self, use_finetuning: bool):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        need_switch = (
            self.current_pipeline is None or
            (self.current_model == "finetuning" and not use_finetuning) or
            (self.current_model == "no_finetuning" and use_finetuning)
        )
        if not need_switch:
            return

        target = "finetuning" if use_finetuning else "no_finetuning"
        mdl_path = (self.model_finetuning_path if use_finetuning
                    else self.model_no_finetuning_path)

        if self.current_pipeline is not None:
            logging.info("Switching pipeline (finetuned = %s)…", use_finetuning)
            del self.current_pipeline; torch.cuda.empty_cache()

        self.current_pipeline = StableDiffusionPipeline.from_pretrained(
            mdl_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
        self.current_model = target

    def predict(self, context, X: Union[pd.DataFrame, dict]) -> pd.DataFrame:
        def _first(val):
            return val.iloc[0] if isinstance(val, pd.Series) else val

        prompt         = _first(X["prompt"])
        use_finetuning = _first(X["use_finetuning"])
        height         = _first(X.get("height", 512))
        width          = _first(X.get("width", 512))
        num_images     = _first(X.get("num_images", 1))
        num_steps      = _first(X.get("num_inference_steps", 100))

        logging.info("Running inference – '%s'", prompt)
        self._load_pipeline(bool(use_finetuning))

        images64: list[str] = []
        with torch.no_grad():
            for i in range(num_images):
                logging.info("Image %d / %d", i + 1, num_images)
                img = self.current_pipeline(prompt, height=height, width=width,
                                            num_inference_steps=num_steps,
                                            guidance_scale=7.5).images[0]
                buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
                images64.append(base64.b64encode(buf.read()).decode())
                img.save(f"local_model_result_{i}.png")

        return pd.DataFrame({"output_images": images64})

   
    @classmethod
    def log_model(cls, finetuned_model_path: str, model_no_finetuning_path: str,
                  artifact_path: str = "image_generation_model"):
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
                "torch", "diffusers", "transformers", "accelerate",
                "pillow", "pandas", "mlflow",
            ],
        )
        logging.info("✅ Model logged to MLflow at '%s'", artifact_path)



def _resolve_accelerate_cfg() -> str:
    base = Path(os.getenv("CONFIG_DIR", "")).expanduser() if os.getenv("CONFIG_DIR") else _find_config_dir()
    n_gpu = torch.cuda.device_count()
    key   = "multi" if n_gpu >= 2 else "single" if n_gpu == 1 else "cpu"
    cfg_path = base / _CONFIG_FILENAMES[key]
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    return str(cfg_path)


def setup_accelerate():
    subprocess.run(["pip", "install", "--quiet", "accelerate"], check=True)
    cfg = _resolve_accelerate_cfg()
    os.environ["ACCELERATE_CONFIG_FILE"] = cfg
    logging.info("Using accelerate cfg: %s", cfg)


def deploy_model():
    setup_accelerate()

    mlflow.set_tracking_uri('/phoenix/mlflow')
    mlflow.set_experiment("ImageGeneration")

    # Use project-relative paths with proper output directory
    project_root = get_project_root()
    finetuned = str(get_output_dir() / "dreambooth")
    
    # Try local model first, fallback to HuggingFace
    local_base_model = project_root / "models" / "stable-diffusion-2-1"
    if local_base_model.exists():
        base = str(local_base_model)
    else:
        # Use HuggingFace model identifier as fallback
        base = "stabilityai/stable-diffusion-2-1"

    # Check if the DreamBooth model exists before proceeding
    if not Path(finetuned).exists():
        logging.warning(f"DreamBooth model not found at {finetuned}")
        logging.warning("Please run DreamBooth training first or use a different finetuned model path.")
        logging.info("Available files in output directory:")
        output_dir = get_output_dir()
        if output_dir.exists():
            for item in os.listdir(output_dir):
                logging.info(f"  - {item}")
        raise FileNotFoundError(f"DreamBooth model not found at {finetuned}")

    logging.info(f"Using finetuned model: {finetuned}")
    logging.info(f"Using base model: {base}")

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
        logging.info("🏷️ Registered 'ImageGenerationService' (run %s)", run.info.run_id)


if __name__ == "__main__":
    deploy_model()
