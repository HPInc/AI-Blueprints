"""
Logger Service implementation for MLflow model logging.

MLflow Registration Layer
- Provides log_model functionality for models
- Handles artifact organization and temporary directory management
- Uses MLflow's models-from-code approach for deployment
- Manages configuration, documents, secrets, and demo assets
"""

import os
import uuid
import base64
import logging
import shutil
from typing import Dict, Any, List
import yaml
import tempfile
import pandas as pd

# Set up logger
logger = logging.getLogger(__name__)


class Logger:
    """
    Logger Service for MLflow model logging.
    This class provides the log_model functionality for packaging RAG-based
    conversational AI with document retrieval capabilities.
    """

    def __init__(self):
        """Initialize the logger service for logging purposes."""
        logger.info("Logger initialized for MLflow model logging")

    @classmethod
    def log_model(
        cls,
        signature,
        artifact_path="AIStudio-Model",
        config_path="configs/config.yaml",
        secrets_dict=None,
        model_path=None,
        demo_folder=None,
    ):
        """
        Log model using MLflow's models-from-code approach.

        Final MLflow structure:
        /artifacts/
          └── data/
              ├── config.yaml
              ├── demo/
              ├── models/
              └── secrets.yaml

        Args:
            signature: MLflow ModelSignature defining input/output schema
            artifact_path: Path to store the model artifacts
            config_path: Path to the configuration file
            secrets_dict: Dict with secrets to persist as YAML (optional)
            model_path: Path to the model file (optional)
            demo_folder: Path to the demo folder (optional)
        """
        import mlflow
        import tempfile
        import shutil
        import os
        import yaml

        # Create temp directory
        temp_base = tempfile.gettempdir()
        temp_dir = os.path.join(temp_base, "model_artifacts")

        # Clean slate for deterministic results
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        try:
            logger.info(f"Organizing artifacts in temp directory: {temp_dir}")

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at: {config_path}")
            shutil.copy2(config_path, os.path.join(temp_dir, "config.yaml"))
            logger.info(f"Copied config from {config_path}")

            if demo_folder and os.path.exists(demo_folder):
                shutil.copytree(demo_folder, os.path.join(temp_dir, "demo"))
                logger.info(f"Copied demo folder")

            if secrets_dict:
                with open(os.path.join(temp_dir, "secrets.yaml"), "w") as f:
                    yaml.safe_dump(secrets_dict, f)
                logger.info("Created secrets.yaml")

            if model_path and os.path.exists(model_path):
                models_temp_dir = os.path.join(temp_dir, "models")
                os.makedirs(models_temp_dir, exist_ok=True)
                if os.path.isfile(model_path):
                    shutil.copy2(
                        model_path,
                        os.path.join(models_temp_dir, os.path.basename(model_path)),
                    )
                    logger.info(f"Copied model file")
                else:
                    shutil.copytree(model_path, models_temp_dir, dirs_exist_ok=True)
                    logger.info(f"Copied model directory")

            mlflow.pyfunc.log_model(
                name=artifact_path,
                loader_module="src.mlflow.loader",
                data_path=temp_dir,
                code_paths=["../src"],
                signature=signature,
                pip_requirements="../requirements.txt",
            )
        except Exception as e:
            logger.error(f"Error during model logging: {str(e)}")
            raise
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary directory")
