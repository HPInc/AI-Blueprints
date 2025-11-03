"""
Logger implementation for MLflow model logging.

MLflow Registration Layer
- Provides log_model functionality for packaging automated evaluation models
- Handles artifact organization and temporary directory management
- Uses MLflow's models-from-code approach for deployment
- Manages configuration, model files, and demo assets
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
    Logger for MLflow model logging.
    This class provides the log_model functionality for packaging automated evaluation models
    with structured outputs capabilities.
    """

    def __init__(self):
        """Initialize the logger for logging purposes."""
        logger.info("Logger initialized for MLflow model logging")

    @classmethod
    def log_model(
        cls,
        signature,
        artifact_path="AIStudio-Evaluation-Model",
        config_path="configs/config.yaml",
        model_path=None,
        demo_folder=None,
    ):
        """
        Log model using refined models-from-code approach with elegant directory structure.

        This implementation uses MLflow's models-from-code approach exclusively with proper
        temp directory naming to avoid redundant nesting while maintaining full MLflow 3.1.0 compatibility.

        Final MLflow structure achieved:
        /artifacts/
          └── data/                    # MLflow automatically created
              ├── config.yaml          # Configuration
              ├── models/              # Model files (optional)
              └── demo/                # UI components (optional)

        Args:
            signature: MLflow ModelSignature defining input/output schema for the model
            artifact_path: Path to store the model artifacts
            config_path: Path to the configuration file
            model_path: Path to the LLaMA model file (optional)
            demo_folder: Path to the demo folder (optional)

        Returns:
            None
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
            # Organize temp directory for clean final structure
            # MLflow will place this under /artifacts/data/ automatically

            # ✅ Config at root -> /artifacts/data/config.yaml
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at: {config_path}")
            shutil.copy2(config_path, os.path.join(temp_dir, "config.yaml"))
            logger.info(f"Copied config from {config_path} to temp directory")

            # ✅ Demo folder -> /artifacts/data/demo/
            if demo_folder and os.path.exists(demo_folder):
                shutil.copytree(demo_folder, os.path.join(temp_dir, "demo"))
                logger.info(f"Copied demo folder from {demo_folder}")
            else:
                logger.info("Demo folder not provided or doesn't exist - skipping")

            # ✅ Handle model files -> /artifacts/data/{model_filename}
            if model_path and os.path.exists(model_path):
                if os.path.isfile(model_path):
                    shutil.copy2(
                        model_path, os.path.join(temp_dir, os.path.basename(model_path))
                    )
                    logger.info(f"Copied model file: {os.path.basename(model_path)}")
                else:
                    # For model directories, copy contents to temp_dir directly
                    for item in os.listdir(model_path):
                        item_path = os.path.join(model_path, item)
                        if os.path.isfile(item_path):
                            shutil.copy2(item_path, temp_dir)
                        else:
                            shutil.copytree(item_path, os.path.join(temp_dir, item))
                    logger.info(f"Copied model directory contents: {model_path}")
            else:
                logger.info("Model path not provided or doesn't exist - skipping")

            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
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
