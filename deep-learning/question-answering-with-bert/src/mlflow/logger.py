"""
Logger Service implementation for MLflow model logging.

MLflow Registration Layer
- Provides log_model functionality for BERT question-answering models
- Handles artifact organization and temporary directory management
- Uses MLflow's models-from-code approach for deployment
- Manages configuration and demo assets
"""

import os
import uuid
import base64
import logging
import shutil
from typing import Dict, Any, List, Optional
import yaml
import tempfile
import pandas as pd

# Set up logger
logger = logging.getLogger(__name__)


class Logger:
    """
    Logger Service for MLflow model logging.
    This class provides the log_model functionality for packaging BERT question-answering models.
    """

    def __init__(self):
        """Initialize the logger service for logging purposes."""
        logger.info("Logger initialized for MLflow BERT QA model logging")

    @classmethod
    def log_model(
        cls,
        signature,
        artifact_path="AIStudio-Model",
        config_path="configs/config.yaml",
        model_checkpoint: str = "distilbert-base-cased",
        source_trainer=None,
        source_pipeline=None,
        demo_folder=None,
    ):
        """
        Log BERT question-answering model using models-from-code approach.

        This implementation uses MLflow's models-from-code approach exclusively with proper
        temp directory naming to avoid redundant nesting while maintaining full MLflow 3.1.0 compatibility.

        Final MLflow structure achieved:
        /artifacts/
          └── data/                    # MLflow automatically created
              ├── config.yaml          # Configuration
              ├── model/               # Trained model directory (optional)
              └── demo/                # UI components (optional)

        Args:
            signature: MLflow ModelSignature defining input/output schema for the model
            artifact_path: Path to store the model artifacts
            config_path: Path to the configuration file
            model_checkpoint: HuggingFace model checkpoint name
            source_trainer: A trainer object with a `.save_model()` method (optional)
            source_pipeline: A pipeline object with a `.save_pretrained()` method (optional)
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
            logger.info(
                f"Organizing BERT QA model artifacts in temp directory: {temp_dir}"
            )
            # Organize temp directory for clean final structure
            # MLflow will place this under /artifacts/data/ automatically

            # ✅ Config at root -> /artifacts/data/config.yaml
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at: {config_path}")
            shutil.copy2(config_path, os.path.join(temp_dir, "config.yaml"))
            logger.info(f"Copied config from {config_path} to temp directory")

            # ✅ Save trained model -> /artifacts/data/model/
            if source_trainer is not None:
                model_temp_dir = os.path.join(temp_dir, "model")
                source_trainer.save_model(model_temp_dir)
                logger.info("Saved model using trainer")
            elif source_pipeline is not None:
                model_temp_dir = os.path.join(temp_dir, "model")
                source_pipeline.model.save_pretrained(model_temp_dir)
                source_pipeline.tokenizer.save_pretrained(model_temp_dir)
                logger.info("Saved model using pipeline")
            else:
                logger.info(
                    "No trainer or pipeline provided - will use default model checkpoint"
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
            logger.info(
                "BERT QA model logged successfully using models-from-code approach"
            )
        except Exception as e:
            logger.error(f"Error during BERT QA model logging: {str(e)}")
            raise
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary directory")
