"""
Simplified image generation service for registration only.
Business logic moved to ImageGenerationModel.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ImageGenerationService:
    """
    Simplified service for model registration only - business logic moved to Model.
    """

    def __init__(self, model_instance=None):
        """Optional model instance for registration purposes."""
        self.model = model_instance

    @classmethod
    def log_model(
        cls, 
        finetuned_model_path: str, 
        model_no_finetuning_path: str,
        artifact_path: str = "image_generation_model",
        config_path: str = "../configs/config.yaml",
        **kwargs
    ):
        """
        MLflow registration using models-from-code approach.
        """
        import mlflow
        from mlflow.types import Schema, ColSpec
        from mlflow.models import ModelSignature
        
        logger.info("Starting model registration using models-from-code approach")
        
        # Define model signature
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

        # Ensure __init__.py files exist for code paths
        core = Path(__file__).resolve().parent.parent
        (core / "__init__.py").touch(exist_ok=True)
        
        # Include both core and src directories, with src at the project root level
        project_root = Path(__file__).resolve().parent.parent.parent
        src_dir = project_root / "src"
        
        # Ensure __init__.py files exist
        (core / "__init__.py").touch(exist_ok=True)
        if src_dir.exists():
            (src_dir / "__init__.py").touch(exist_ok=True)

        # Prepare artifacts
        artifacts = {
            "finetuned_model": finetuned_model_path,
            "model_no_finetuning": model_no_finetuning_path,
        }
        
        # Add config if it exists
        if config_path and os.path.exists(str(Path(config_path).resolve())):
            artifacts["config"] = str(Path(config_path).resolve())

        # Prepare code paths
        code_paths = [str(core)]
        if src_dir.exists():
            code_paths.append(str(src_dir))

        # Use models-from-code approach instead of pyfunc.log_model
        mlflow.models.log_model(
            artifact_path=artifact_path,
            loader_module="core.image_generation_service.image_generation_loader",
            data_path=None,
            artifacts=artifacts,
            signature=signature,
            code_paths=code_paths,
            pip_requirements="../requirements.txt"
        )
        
        logger.info("✅ Model logged to MLflow at '%s' using models-from-code", artifact_path)

    @classmethod 
    def log_model_metadata(cls, artifacts: Dict[str, Any]):
        """Log only model metadata without copying full models for faster deployment"""
        import mlflow
        
        # Log only essential metadata for faster deployment
        mlflow.log_params({
            "model_type": "stable_diffusion_2_1",
            "finetuned_model_path": artifacts.get("finetuned_model", ""),
            "base_model_path": artifacts.get("model_no_finetuning", ""),
            "model_architecture": "stable_diffusion",
            "memory_efficient": True
        })
        logger.info("✅ Model metadata logged to MLflow")
