

from __future__ import annotations

import base64, io, logging, os, shutil, subprocess, sys, time, gc
from pathlib import Path
from typing import Union

import mlflow, torch, pandas as pd, numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from mlflow.types import Schema, ColSpec
from mlflow.models import ModelSignature

# Check for xformers availability for memory-efficient attention
try:
    import xformers
    _XFORMERS_AVAILABLE = True
except ImportError:
    _XFORMERS_AVAILABLE = False

# Import utility functions from src
# Add the project root to the path for proper src module import resolution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import get_project_root, get_config_dir, get_output_dir

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
        logging.info("Loading model artifacts (lazy loading enabled)…")
        # Store paths only, don't load models yet for memory efficiency
        self.model_no_finetuning_path = context.artifacts["model_no_finetuning"]
        self.model_finetuning_path    = context.artifacts["finetuned_model"]

        # Validate model paths and provide helpful information
        self._validate_model_paths()

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus >= 2:
            logging.info("Detected %d GPUs (multi-GPU pipeline)", self.num_gpus)
        elif self.num_gpus == 1:
            logging.info("Detected 1 GPU (single-GPU pipeline)")
        else:
            logging.info("Running on CPU")
        
        # Initialize as None for lazy loading
        self.current_pipeline, self.current_model = None, None
        
        # Clear GPU memory at initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def _validate_model_paths(self):
        """Validate and log information about model paths"""
        base_path = Path(self.model_no_finetuning_path)
        finetuned_path = Path(self.model_finetuning_path)
        
        logging.info(f"Base model path: {self.model_no_finetuning_path}")
        logging.info(f"Fine-tuned model path: {self.model_finetuning_path}")
        
        # Check if paths are local directories or HuggingFace model identifiers
        if base_path.exists():
            logging.info("✅ Base model found locally")
        else:
            logging.info("🔄 Base model will be downloaded from HuggingFace Hub")
            
        if finetuned_path.exists():
            logging.info("✅ Fine-tuned model found locally")
            # Check for common fp16 variant files
            fp16_files = list(finetuned_path.glob("*fp16*"))
            if fp16_files:
                logging.info(f"📁 Found {len(fp16_files)} fp16 variant files in fine-tuned model")
            else:
                logging.warning("⚠️  No fp16 variant files found in fine-tuned model - will use standard loading")
        else:
            logging.warning("⚠️  Fine-tuned model path does not exist locally")
            logging.warning("    This may cause errors when use_finetuning=true")

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

        # Memory cleanup before loading new pipeline
        if self.current_pipeline is not None:
            logging.info("Switching pipeline (finetuned = %s)…", use_finetuning)
            del self.current_pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Load pipeline with model-specific handling
        logging.info(f"Loading {'fine-tuned' if use_finetuning else 'base'} model from: {mdl_path}")
        self.current_pipeline = self._load_model_with_fallbacks(mdl_path, device, use_finetuning)
        
        # Apply memory-efficient setup
        self._setup_pipeline(self.current_pipeline)
        self.current_model = target
        
        logging.info("Pipeline loaded for %s", target)

    def _load_model_with_fallbacks(self, mdl_path: str, device: str, use_finetuning: bool):
        """
        Load model with targeted fallback strategies based on model type.
        Fine-tuned models are more likely to lack fp16 variants than base models.
        """
        model_type = "fine-tuned" if use_finetuning else "base"
        
        # Strategy 1: Try optimal configuration (fp16 variant + fp16 dtype)
        if torch.cuda.is_available():
            try:
                logging.info(f"Attempting to load {model_type} model with fp16 variant and dtype")
                return StableDiffusionPipeline.from_pretrained(
                    mdl_path, 
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    variant="fp16"
                ).to(device)
            except ValueError as e:
                if "variant=fp16" in str(e):
                    logging.warning(f"{model_type.title()} model lacks fp16 variant - this is common for fine-tuned models")
                else:
                    logging.warning(f"Failed to load {model_type} model with fp16 variant: {e}")
            except Exception as e:
                logging.warning(f"Unexpected error loading {model_type} model with fp16 variant: {e}")
        
        # Strategy 2: Try fp16 dtype without variant (most common fallback)
        try:
            logging.info(f"Loading {model_type} model with fp16 dtype (no variant)")
            return StableDiffusionPipeline.from_pretrained(
                mdl_path, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            ).to(device)
        except Exception as e:
            logging.warning(f"Failed to load {model_type} model with fp16 dtype: {e}")
        
        # Strategy 3: Try fp32 dtype (compatibility fallback)
        try:
            logging.info(f"Loading {model_type} model with fp32 dtype (compatibility mode)")
            return StableDiffusionPipeline.from_pretrained(
                mdl_path, 
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(device)
        except Exception as e:
            logging.warning(f"Failed to load {model_type} model with fp32 dtype: {e}")
        
        # Strategy 4: Minimal configuration (last resort)
        try:
            logging.warning(f"Loading {model_type} model with minimal configuration (last resort)")
            return StableDiffusionPipeline.from_pretrained(
                mdl_path,
                low_cpu_mem_usage=True
            ).to(device)
        except Exception as e:
            logging.error(f"Failed to load {model_type} model even with minimal configuration: {e}")
            raise RuntimeError(f"Unable to load {model_type} model from {mdl_path}. "
                             f"Please check if the model path is valid and accessible.") from e

    def _setup_pipeline(self, pipeline):
        """Apply memory-efficient setup to the pipeline"""
        try:
            # Enable memory-efficient attention if xformers is available
            if _XFORMERS_AVAILABLE and hasattr(pipeline, 'unet'):
                pipeline.unet.enable_xformers_memory_efficient_attention()
                logging.info("Enabled xformers memory-efficient attention")
        except Exception as e:
            logging.warning("Could not enable xformers attention: %s", e)
        
        try:
            # Enable attention slicing for memory efficiency
            if hasattr(pipeline, 'enable_attention_slicing'):
                pipeline.enable_attention_slicing(slice_size="auto")
                logging.info("Enabled attention slicing")
        except Exception as e:
            logging.warning("Could not enable attention slicing: %s", e)
            
        try:
            # Enable CPU offloading for large models if needed
            if torch.cuda.is_available() and hasattr(pipeline, 'enable_sequential_cpu_offload'):
                # Only enable if we have limited GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                if gpu_memory < 12:  # For GPUs with less than 12GB
                    pipeline.enable_sequential_cpu_offload()
                    logging.info("Enabled sequential CPU offloading for memory management")
        except Exception as e:
            logging.warning("Could not enable CPU offloading: %s", e)

    def predict(self, context, X: Union[pd.DataFrame, dict]) -> pd.DataFrame:
        def _first(val):
            return val.iloc[0] if isinstance(val, pd.Series) else val

        prompt         = _first(X["prompt"])
        use_finetuning = _first(X["use_finetuning"])
        height         = _first(X.get("height", 512))
        width          = _first(X.get("width", 512))
        # Keep original defaults to maintain image quality
        num_images     = _first(X.get("num_images", 1))
        num_steps      = _first(X.get("num_inference_steps", 100))

        logging.info("Running inference – '%s'", prompt)
        self._load_pipeline(bool(use_finetuning))

        images64: list[str] = []
        with torch.no_grad():
            for i in range(num_images):
                logging.info("Image %d / %d", i + 1, num_images)
                # Use original inference parameters for quality
                img = self.current_pipeline(
                    prompt, 
                    height=height, 
                    width=width,
                    num_inference_steps=num_steps
                ).images[0]
                
                buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
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

   
    @classmethod
    def log_model(
        cls, 
        finetuned_model_path: str, 
        model_no_finetuning_path: str,
        artifact_path: str = "image_generation_model",
        config_path: str = "../configs/config.yaml",
        demo_path: str = "../demo"
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
        
        # Include both core and src directories, with src at the project root level
        project_root = Path(__file__).resolve().parent.parent.parent
        src_dir = project_root / "src"
        
        # Ensure __init__.py files exist
        (core / "__init__.py").touch(exist_ok=True)
        (src_dir / "__init__.py").touch(exist_ok=True)

        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=cls(),
            artifacts={
                "finetuned_model":     finetuned_model_path,
                "model_no_finetuning": model_no_finetuning_path,
                "config": str(Path(config_path).resolve()),
                "demo": str(Path(demo_path))
            },
            signature=signature,
            code_paths=[str(core), str(src_dir)],
            pip_requirements="../requirements.txt",
        )
        logging.info("✅ Model logged to MLflow at '%s'", artifact_path)

    @classmethod
    def log_model_metadata(cls, artifacts: dict):
        """Log only model metadata without copying full models for faster deployment"""
        # Log only essential metadata for faster deployment
        mlflow.log_params({
            "model_type": "stable_diffusion_2_1",
            "finetuned_model_path": artifacts.get("finetuned_model", ""),
            "base_model_path": artifacts.get("model_no_finetuning", ""),
            "model_type": "stable_diffusion",
            "memory_efficient": True
        })
        logging.info("✅ Model metadata logged to MLflow")



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
    try:
        setup_accelerate()

        # Pre-deployment memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logging.info("Starting model deployment...")

        mlflow.set_tracking_uri('/phoenix/mlflow')
        mlflow.set_experiment("ImageGeneration")

        # Use project-relative paths with proper output directory
        project_root = get_project_root()
        finetuned = str(get_output_dir() / "dreambooth")
        
        # Try local model first, fallback to HuggingFace
        local_base_model = project_root / "models" / "stable-diffusion-2-1"
        if local_base_model.exists():
            base = str(local_base_model)
            logging.info("Using local base model: %s", base)
        else:
            # Use HuggingFace model identifier as fallback
            base = "stabilityai/stable-diffusion-2-1"
            logging.info("Using HuggingFace base model: %s", base)

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
            logging.info("📦 Logging artifacts and model...")
            
            # Log only accelerate config without loading models
            mlflow.log_artifact(os.environ["ACCELERATE_CONFIG_FILE"],
                                artifact_path="accelerate_config")

            # Log model with configuration
            ImageGenerationModel.log_model(
                finetuned_model_path=finetuned,
                model_no_finetuning_path=base,
                config_path="../configs/config.yaml",
                demo_path = "../demo"
            )
            
            # Post-deployment cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            model_uri = f"runs:/{run.info.run_id}/image_generation_model"
            mlflow.register_model(model_uri=model_uri,
                                  name="ImageGenerationService")
            logging.info("🏷️ Registered 'ImageGenerationService' (run %s)", run.info.run_id)
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
