"""
Logger Service implementation for MLflow model logging.

MLflow Registration Layer for RNN Text Generation
- Provides log_model functionality for RNN text generation models
- Handles RNN-specific artifact organization and temporary directory management
- Uses MLflow's models-from-code approach for deployment
- Manages model state dictionaries, encoders, decoders, configuration, and demo assets
"""

import os
import sys
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
    This class provides the log_model functionality for packaging RNN-based
    text generation models with character-level processing.
    """

    def __init__(self):
        """Initialize the logger service for logging purposes."""
        logger.info("Logger initialized for MLflow RNN model logging")

    @classmethod
    def log_model(
        cls,
        signature,
        model_state_dict_path,
        decoder_path,
        encoder_path,
        artifact_path="AIStudio-Model",
        config_path="configs/config.yaml",
        data_path=None,
        demo_folder=None,
        docs_path="data/",
    ):
        """
        Log RNN text generation model using models-from-code approach.

        This implementation uses MLflow's models-from-code approach exclusively with proper
        temp directory naming to avoid redundant nesting while maintaining full MLflow 3.1.0 compatibility.

        Final MLflow structure achieved:
        /artifacts/
          └── data/                    # MLflow automatically created
              ├── config.yaml          # Configuration
              ├── model_state_dict     # Trained RNN model
              ├── decoder              # Character decoder dictionary
              ├── encoder              # Character encoder dictionary
              ├── shakespeare.txt      # Training data (optional)
              └── demo/                # UI components (optional)

        Args:
            signature: MLflow ModelSignature defining input/output schema for the model
            model_state_dict_path: Path to the trained model state dictionary
            decoder_path: Path to the character decoder dictionary
            encoder_path: Path to the character encoder dictionary
            artifact_path: Path to store the model artifacts
            config_path: Path to the configuration file
            data_path: Path to the training data file (optional)
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
            logger.info(f"Organizing RNN artifacts in temp directory: {temp_dir}")

            # ✅ Config at root -> /artifacts/data/config.yaml
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at: {config_path}")
            shutil.copy2(config_path, os.path.join(temp_dir, "config.yaml"))
            logger.info(f"Copied config from {config_path} to temp directory")

            data_temp_dir = os.path.join(temp_dir, "data")
            os.makedirs(data_temp_dir, exist_ok=True)

            if docs_path and os.path.exists(docs_path):
                for item in os.listdir(docs_path):
                    item_path = os.path.join(docs_path, item)
                    if os.path.isfile(item_path):
                        shutil.copy2(item_path, data_temp_dir)
                        logger.info(f"Copied document: {item}")
                    elif os.path.isdir(item_path):
                        shutil.copytree(item_path, os.path.join(data_temp_dir, item))
                        logger.info(f"Copied document directory: {item}")
            logger.info("data folder not provided or doesn't exist - skipping")

            # ✅ Required RNN model artifacts
            if not os.path.exists(model_state_dict_path):
                raise FileNotFoundError(
                    f"Model state dict not found at: {model_state_dict_path}"
                )
            shutil.copy2(
                model_state_dict_path, os.path.join(temp_dir, "model_state_dict")
            )
            logger.info(f"Copied model state dict from {model_state_dict_path}")

            if not os.path.exists(decoder_path):
                raise FileNotFoundError(f"Decoder not found at: {decoder_path}")
            shutil.copy2(decoder_path, os.path.join(temp_dir, "decoder"))
            logger.info(f"Copied decoder from {decoder_path}")

            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Encoder not found at: {encoder_path}")
            shutil.copy2(encoder_path, os.path.join(temp_dir, "encoder"))
            logger.info(f"Copied encoder from {encoder_path}")

            # ✅ Optional training data file -> /artifacts/data/shakespeare.txt
            if data_path and os.path.exists(data_path):
                shutil.copy2(data_path, os.path.join(temp_dir, "shakespeare.txt"))
                logger.info(f"Copied training data from {data_path}")
            else:
                logger.info("Training data not provided or doesn't exist - skipping")

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
            logger.info("RNN Model logged to MLflow successfully")
        except Exception as e:
            logger.error(f"Error during RNN model logging: {str(e)}")
            raise
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary directory")
