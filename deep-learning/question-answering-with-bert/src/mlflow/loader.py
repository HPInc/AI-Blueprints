"""
MLflow models-from-code loader module for BERT Question Answering.
This module provides the _load_pyfunc function required by MLflow's models-from-code approach.
"""

import os
import logging
from typing import Dict, Any, Optional

# Set up logger
logger = logging.getLogger(__name__)


def _load_pyfunc(data_path: str):
    """
    MLflow models-from-code loader function.
    Called by MLflow to load the BERT question-answering model from artifacts.

    Args:
        data_path: Path to model artifacts directory containing:
            - config.yaml: Model configuration
            - model/: Model files directory (for the trained BERT model)
            - demo/: Demo folder with UI components (optional)

    Returns:
        Model: Initialized BERT question-answering model instance ready for prediction
    """
    from src.mlflow.model import Model

    logger.info(f"Loading BERT QA Model from artifacts at: {data_path}")

    from src.utils import load_config

    config_path = os.path.join(data_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    config = load_config(config_path)
    logger.info("Configuration loaded successfully")

    # Get model checkpoint from artifacts - check if we have a saved model
    model_dir = os.path.join(data_path, "model")
    if os.path.exists(model_dir):
        # Use the saved model from artifacts
        model_checkpoint = model_dir
        logger.info(f"Using saved model from artifacts: {model_checkpoint}")
    else:
        # Fall back to default model checkpoint
        model_checkpoint = "distilbert-base-cased"
        logger.info(f"Using default model checkpoint: {model_checkpoint}")

    # Initialize Model
    try:
        model = Model(model_checkpoint=model_checkpoint, config=config)
        logger.info("BERT QA Model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize BERT QA Model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e
