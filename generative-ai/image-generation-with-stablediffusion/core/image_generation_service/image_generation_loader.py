"""
MLflow models-from-code loader for image generation service.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any

from .image_generation_model import ImageGenerationModel

logger = logging.getLogger(__name__)


def _load_model(model_uri: str):
    """
    Load model components and return configured ImageGenerationModel instance.
    
    Args:
        model_uri: URI of the MLflow model
        
    Returns:
        Configured ImageGenerationModel instance
    """
    logger.info("Loading image generation model from URI: %s", model_uri)
    
    # Extract model artifacts path from environment
    artifacts_path = os.environ.get("MODEL_ARTIFACTS_PATH", "")
    
    if not artifacts_path:
        raise ValueError("MODEL_ARTIFACTS_PATH environment variable not set")
    
    # Construct paths to model artifacts
    model_no_finetuning_path = os.path.join(artifacts_path, "model_no_finetuning")
    model_finetuning_path = os.path.join(artifacts_path, "finetuned_model")
    config_path = os.path.join(artifacts_path, "config")
    
    # Load configuration if available
    config = {}
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded from artifacts")
    else:
        logger.warning("No configuration found in artifacts")
    
    # Validate model paths exist
    base_path = Path(model_no_finetuning_path)
    finetuned_path = Path(model_finetuning_path)
    
    if not base_path.exists() and not base_path.name.startswith("stabilityai"):
        logger.warning(f"Base model path does not exist: {model_no_finetuning_path}")
    
    if not finetuned_path.exists():
        logger.warning(f"Fine-tuned model path does not exist: {model_finetuning_path}")
    
    # Create and return Model instance (not Service!)
    model = ImageGenerationModel(
        model_no_finetuning_path=model_no_finetuning_path,
        model_finetuning_path=model_finetuning_path,
        config=config
    )
    
    logger.info("Image generation model loaded successfully")
    return model


def save_model(
    model_no_finetuning_path: str,
    model_finetuning_path: str,
    model_path: str,
    config_path: str = None,
    sample_input: Dict[str, Any] = None,
    sample_output: Dict[str, Any] = None
):
    """
    Save model using models-from-code approach.
    
    Args:
        model_no_finetuning_path: Path to base model
        model_finetuning_path: Path to fine-tuned model
        model_path: Path to save the MLflow model
        config_path: Path to configuration file
        sample_input: Sample input for signature inference
        sample_output: Sample output for signature inference
    """
    import mlflow
    from mlflow.models import infer_signature
    from mlflow.types import Schema, ColSpec
    from mlflow.models import ModelSignature
    
    # Define input/output schema for image generation
    input_schema = Schema([
        ColSpec("string", "prompt"),
        ColSpec("boolean", "use_finetuning"),
        ColSpec("integer", "height"),
        ColSpec("integer", "width"),
        ColSpec("integer", "num_images"),
        ColSpec("integer", "num_inference_steps"),
    ])
    output_schema = Schema([ColSpec("string", "output_images")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    
    # Prepare artifacts
    artifacts = {
        "model_no_finetuning": model_no_finetuning_path,
        "finetuned_model": model_finetuning_path,
    }
    
    if config_path and os.path.exists(config_path):
        artifacts["config"] = config_path
    
    # Prepare code paths
    core_path = str(Path(__file__).parent.parent)
    project_root = str(Path(__file__).parent.parent.parent)
    src_path = os.path.join(project_root, "src")
    
    code_paths = [core_path]
    if os.path.exists(src_path):
        code_paths.append(src_path)
    
    # Save model using models-from-code
    mlflow.models.save_model(
        path=model_path,
        loader_module="image_generation_loader",
        data_path=None,
        signature=signature,
        artifacts=artifacts,
        code_paths=code_paths,
        pip_requirements=os.path.join(project_root, "requirements.txt") if os.path.exists(os.path.join(project_root, "requirements.txt")) else None
    )
    
    logger.info("✅ Model saved using models-from-code approach at: %s", model_path)
