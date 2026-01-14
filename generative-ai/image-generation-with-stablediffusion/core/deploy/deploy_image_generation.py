from __future__ import annotations

import gc
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
import mlflow
import torch

# Import utility functions from src
# Add the project root to the path for proper src module import resolution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import get_project_root, get_config_dir, get_output_dir

# Import the new service
from src.mlflow.logger import Logger

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)


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


def _resolve_accelerate_cfg() -> str:
    base = (
        Path(os.getenv("CONFIG_DIR", "")).expanduser()
        if os.getenv("CONFIG_DIR")
        else _find_config_dir()
    )
    n_gpu = torch.cuda.device_count()
    key = "multi" if n_gpu >= 2 else "single" if n_gpu == 1 else "cpu"
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
    try:
        setup_accelerate()

        # Pre-deployment memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        logging.info("Starting model deployment...")

        mlflow.set_tracking_uri("/phoenix/mlflow")
        mlflow.set_experiment("ImageGeneration")

        # Use project-relative paths with proper output directory
        project_root = get_project_root()
        finetuned = str(get_output_dir() / "dreambooth")

        # Try local model first, fallback to HuggingFace
        local_base_model = project_root / "models" / "stable-diffusion-2-1-base"
        if local_base_model.exists():
            base = str(local_base_model)
            logging.info("Using local base model: %s", base)
        else:
            # Use HuggingFace model identifier as fallback
            base = "Manojb/stable-diffusion-2-1-base"
            logging.info("Using HuggingFace base model: %s", base)

        # Check if the DreamBooth model exists before proceeding
        if not Path(finetuned).exists():
            logging.warning(f"DreamBooth model not found at {finetuned}")
            logging.warning(
                "Please run DreamBooth training first or use a different finetuned model path."
            )
            logging.info("Available files in output directory:")
            output_dir = get_output_dir()
            if output_dir.exists():
                for item in os.listdir(output_dir):
                    logging.info(f"  - {item}")
            raise FileNotFoundError(f"DreamBooth model not found at {finetuned}")

        logging.info(f"Using finetuned model: {finetuned}")
        logging.info(f"Using base model: {base}")

        with mlflow.start_run(run_name="image_generation_service") as run:
            logging.info("📦 Logging artifacts and model...")

            # Log only accelerate config without loading models
            mlflow.log_artifact(
                os.environ["ACCELERATE_CONFIG_FILE"], artifact_path="accelerate_config"
            )

            # Create signature for image generation model
            from mlflow.models.signature import ModelSignature
            from mlflow.types.schema import Schema, ColSpec

            input_schema = Schema(
                [
                    ColSpec("string", "prompt"),
                    ColSpec("boolean", "use_finetuning"),
                    ColSpec("integer", "height"),
                    ColSpec("integer", "width"),
                    ColSpec("integer", "num_images"),
                    ColSpec("integer", "num_inference_steps"),
                ]
            )
            output_schema = Schema([ColSpec("string", "output_images")])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            # Log model using new models-from-code service
            Logger.log_model(
                signature=signature,
                artifact_path="image_generation_model",
                config_path="../configs/config.yaml",
                model_no_finetuning_path=base,
                model_finetuning_path=finetuned,
                demo_folder="../demo",
            )

            # Post-deployment cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            model_uri = f"runs:/{run.info.run_id}/image_generation_model"
            mlflow.register_model(model_uri=model_uri, name="ImageGenerationLogger")
            logging.info(
                "🏷️ Registered 'ImageGenerationLogger' (run %s)", run.info.run_id
            )
            logging.info("Model deployment completed successfully")

    except Exception as e:
        logging.error(f"❌ Model deployment failed: {str(e)}")
        # Cleanup on failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise


if __name__ == "__main__":
    deploy_model()
