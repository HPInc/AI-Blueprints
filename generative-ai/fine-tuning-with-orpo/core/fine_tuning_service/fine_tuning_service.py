"""
Fine-Tuning Service implementation for MLflow model logging.

MLflow Registration Layer
- Provides log_model functionality for packaging fine-tuning comparison models
- Handles artifact organization and temporary directory management
- Uses MLflow's models-from-code approach for deployment
- Manages configuration, model paths, secrets, and demo assets
"""

import os
import logging
import shutil
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path

# Set up logger
logger = logging.getLogger(__name__)


class FineTuningService:
    """
    Fine-Tuning Service for MLflow model logging.
    This class provides the log_model functionality for packaging LLM comparison
    models that can switch between base and fine-tuned versions.
    """

    def __init__(self):
        """Initialize the fine-tuning service for logging purposes."""
        logger.info("FineTuningService initialized for MLflow model logging")

    @classmethod
    def log_model(
        cls,
        artifact_path="llm_serving_model",
        config_path="../configs/config.yaml",
        base_model_path="",
        finetuned_model_path="",
        secrets_dict=None,
        demo_folder="../demo",
    ):
        """
        Log model using models-from-code approach with proper artifact organization.

        This implementation uses MLflow's models-from-code approach to eliminate
        serialization issues while maintaining full MLflow 3.1.0 compatibility.

        Final MLflow structure achieved:
        /artifacts/
          └── data/                           # MLflow automatically created
              ├── config.yaml                 # Configuration
              ├── model_no_finetuning/        # Base model directory or HF ID file
              ├── finetuned_model/            # Fine-tuned model directory or HF ID file
              ├── demo/                       # UI components
              └── secrets.yaml                # Secrets (optional)

        Args:
            artifact_path: Path to store the model artifacts
            config_path: Path to the configuration file
            base_model_path: Path to base model (directory or HF model ID)
            finetuned_model_path: Path to fine-tuned model (directory or HF model ID)
            secrets_dict: Dict with secrets to persist as YAML (optional)
            demo_folder: Path to the demo folder (optional)

        Returns:
            None
        """
        import mlflow
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec
        import yaml

        # Define model input/output schema
        input_schema = Schema(
            [
                ColSpec("string", "prompt"),
                ColSpec("boolean", "use_finetuning"),
                ColSpec("integer", "max_tokens"),
            ]
        )
        output_schema = Schema([ColSpec("string", "response")])

        # Create signature
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # Create temp directory
        temp_base = tempfile.gettempdir()
        temp_dir = os.path.join(temp_base, "fine_tuning_model_artifacts")

        # Clean slate for deterministic results
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        try:
            logger.info(f"Organizing artifacts in temp directory: {temp_dir}")

            # ✅ Config at root -> /artifacts/data/config.yaml
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at: {config_path}")
            shutil.copy2(config_path, os.path.join(temp_dir, "config.yaml"))
            logger.info(f"Copied config from {config_path} to temp directory")

            # ✅ Handle base model -> /artifacts/data/model_no_finetuning/
            base_artifact_dir = os.path.join(temp_dir, "model_no_finetuning")
            cls._handle_model_artifact(base_model_path, base_artifact_dir, "base")

            # ✅ Handle fine-tuned model -> /artifacts/data/finetuned_model/
            ft_artifact_dir = os.path.join(temp_dir, "finetuned_model")
            cls._handle_model_artifact(
                finetuned_model_path, ft_artifact_dir, "fine-tuned"
            )

            # ✅ Demo folder -> /artifacts/data/demo/
            if demo_folder and os.path.exists(demo_folder):
                shutil.copytree(demo_folder, os.path.join(temp_dir, "demo"))
                logger.info(f"Copied demo folder from {demo_folder}")
            else:
                logger.info("Demo folder not provided or doesn't exist - skipping")

            # ✅ Handle secrets -> /artifacts/data/secrets.yaml
            if secrets_dict:
                with open(os.path.join(temp_dir, "secrets.yaml"), "w") as f:
                    yaml.safe_dump(secrets_dict, f)
                logger.info("Created secrets.yaml in temp directory")

            # Log model using models-from-code approach
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                loader_module="core.fine_tuning_service.fine_tuning_loader",
                data_path=temp_dir,
                code_paths=["../core", "../src"],
                signature=signature,
                pip_requirements="../requirements.txt",
            )

            logger.info(
                f"Model logged successfully with artifact path: {artifact_path}"
            )

        except Exception as e:
            logger.error(f"Error during model logging: {str(e)}")
            raise
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary directory")

    @staticmethod
    def _handle_model_artifact(model_path: str, artifact_dir: str, model_type: str):
        """
        Handle model artifact creation - either copy directory or save HF model ID.

        Args:
            model_path: Source model path (directory or HF model ID)
            artifact_dir: Destination artifact directory
            model_type: Type of model for logging (base/fine-tuned)
        """
        model_path_obj = Path(model_path)

        if model_path_obj.exists() and model_path_obj.is_dir():
            # It's a local directory - copy it
            shutil.copytree(model_path, artifact_dir)
            logger.info(f"Copied {model_type} model directory: {model_path}")
        elif model_path_obj.exists() and model_path_obj.is_file():
            # It's a file - copy it
            os.makedirs(artifact_dir, exist_ok=True)
            shutil.copy2(
                model_path, os.path.join(artifact_dir, os.path.basename(model_path))
            )
            logger.info(f"Copied {model_type} model file: {model_path}")
        else:
            # Assume it's a HuggingFace model ID or path - save it as a reference
            os.makedirs(artifact_dir, exist_ok=True)
            with open(os.path.join(artifact_dir, "model_id.txt"), "w") as f:
                f.write(model_path)
            logger.info(f"Saved {model_type} model ID reference: {model_path}")

    @classmethod
    def register_model(
        cls,
        model_base_path: str,
        model_finetuned_path: str,
        experiment: str,
        run_name: str,
        registry_name: str,
        config_path: str = "../configs/config.yaml",
        demo_folder: str = "../demo",
    ):
        """
        Register a fine-tuning comparison model with MLflow.
        This is a wrapper around the original register_llm_comparison_model function
        that uses the new models-from-code approach.

        Args:
            model_base_path: Path to base model (can be relative to project)
            model_finetuned_path: Path to fine-tuned model (can be relative to project)
            experiment: MLflow experiment name
            run_name: MLflow run name
            registry_name: Model registry name
            config_path: Path to configuration file
            demo_folder: Path to demo folder
        """
        import mlflow

        # Resolve model paths using project utilities
        def resolve_model_path(path_str: str) -> str:
            """Resolve model path, making it project-relative if needed."""
            from src.utils import (
                get_project_root,
                get_models_dir,
                get_fine_tuned_models_dir,
            )

            path = Path(path_str)

            # If absolute path and exists, use as-is
            if path.is_absolute() and path.exists():
                return str(path)

            # If relative path, try to resolve relative to project directories
            project_root = get_project_root()

            # Try models directory
            models_path = get_models_dir() / path_str
            if models_path.exists():
                return str(models_path)

            # Try fine-tuned models directory
            ft_path = get_fine_tuned_models_dir() / path_str
            if ft_path.exists():
                return str(ft_path)

            # Try relative to project root
            root_path = project_root / path_str
            if root_path.exists():
                return str(root_path)

            # If it's a HuggingFace model ID, return as-is
            if "/" in path_str and not path_str.startswith("../"):
                return path_str

            # Return original path and let downstream handle the error
            logger.warning(f"Could not resolve model path: {path_str}")
            return path_str

        resolved_base_path = resolve_model_path(model_base_path)
        resolved_ft_path = resolve_model_path(model_finetuned_path)

        logger.info(f"Resolved base model path: {resolved_base_path}")
        logger.info(f"Resolved fine-tuned model path: {resolved_ft_path}")

        # Set experiment and start run
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as run:
            # Log model using new approach
            cls.log_model(
                artifact_path="llm_serving_model",
                config_path=config_path,
                base_model_path=resolved_base_path,
                finetuned_model_path=resolved_ft_path,
                demo_folder=demo_folder,
            )

            # Register model
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/llm_serving_model",
                name=registry_name,
            )

            logger.info(
                "✅ Fine-tuning comparison model registered as `%s` (run %s)",
                registry_name,
                run.info.run_id,
            )
