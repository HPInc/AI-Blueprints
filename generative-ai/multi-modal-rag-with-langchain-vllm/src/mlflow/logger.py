"""
Logger Service implementation for MLflow model logging.

MLflow Registration Layer
- Provides log_model functionality for models
- Handles artifact organization and temporary directory management
- Uses MLflow's models-from-code approach for deployment
- Manages configuration, model directories, and demo assets
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
    This class provides the log_model functionality for packaging multimodal RAG-based
    conversational AI with document retrieval capabilities.
    """

    def __init__(self):
        """Initialize the logger service for logging purposes."""
        logger.info("Logger initialized for MLflow model logging")

    @classmethod
    def log_model(
        cls,
        artifact_path="AIStudio-Model",
        config_path="configs/config.yaml",
        local_model_dir=None,
        e5_model_dir=None,
        siglip_model_dir=None,
        demo_folder=None,
        signature=None,
    ):
        """
        Log model using refined models-from-code approach with elegant directory structure.

        This implementation uses MLflow's models-from-code approach exclusively with proper
        temp directory naming to avoid redundant nesting while maintaining full MLflow 3.1.0 compatibility.

        Final MLflow structure achieved:
        /artifacts/
          └── data/                    # MLflow automatically created
              ├── config.yaml          # Configuration
              ├── local_model_dir/     # Local Qwen-VL model directory
              ├── e5_model_dir/        # E5 embedding model directory
              ├── siglip_model_dir/    # SigLIP image embedding model directory
              └── demo/                # UI components (optional)

        Args:
            artifact_path: Path to store the model artifacts
            config_path: Path to the configuration file
            local_model_dir: Path to the local Qwen-VL model directory
            e5_model_dir: Path to the E5 embedding model directory
            siglip_model_dir: Path to the SigLIP image embedding model directory
            demo_folder: Path to the demo folder (optional)
            signature: MLflow ModelSignature defining input/output schema for the model

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

            # ✅ Local model directory -> /artifacts/data/local_model_dir/
            if local_model_dir and os.path.exists(local_model_dir):
                local_temp_dir = os.path.join(temp_dir, "local_model_dir")
                if os.path.isfile(local_model_dir):
                    # If it's a single file, create directory and copy file
                    os.makedirs(local_temp_dir, exist_ok=True)
                    shutil.copy2(
                        local_model_dir,
                        os.path.join(local_temp_dir, os.path.basename(local_model_dir)),
                    )
                    logger.info(
                        f"Copied local model file: {os.path.basename(local_model_dir)}"
                    )
                else:
                    # If it's a directory, copy entire directory
                    shutil.copytree(local_model_dir, local_temp_dir)
                    logger.info(f"Copied local model directory: {local_model_dir}")
            else:
                raise FileNotFoundError(
                    f"Local model directory not found at: {local_model_dir}"
                )

            # ✅ E5 model directory -> /artifacts/data/e5_model_dir/
            if e5_model_dir and os.path.exists(e5_model_dir):
                e5_temp_dir = os.path.join(temp_dir, "e5_model_dir")
                shutil.copytree(e5_model_dir, e5_temp_dir)
                logger.info(f"Copied E5 model directory: {e5_model_dir}")
            else:
                raise FileNotFoundError(
                    f"E5 model directory not found at: {e5_model_dir}"
                )

            # ✅ SigLIP model directory -> /artifacts/data/siglip_model_dir/
            if siglip_model_dir and os.path.exists(siglip_model_dir):
                siglip_temp_dir = os.path.join(temp_dir, "siglip_model_dir")
                shutil.copytree(siglip_model_dir, siglip_temp_dir)
                logger.info(f"Copied SigLIP model directory: {siglip_model_dir}")
            else:
                raise FileNotFoundError(
                    f"SigLIP model directory not found at: {siglip_model_dir}"
                )

            # ✅ Demo folder -> /artifacts/data/demo/
            if demo_folder and os.path.exists(demo_folder):
                shutil.copytree(demo_folder, os.path.join(temp_dir, "demo"))
                logger.info(f"Copied demo folder from {demo_folder}")
            else:
                logger.info("Demo folder not provided or doesn't exist - skipping")

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
