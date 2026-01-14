"""
MLflow models-from-code loader module for image generation.
This module provides the _load_pyfunc function required by MLflow's.
"""

import os
import logging
from typing import Dict, Any, Optional

# Set up logger
logger = logging.getLogger(__name__)


def _load_pyfunc(data_path: str):
    """
    MLflow models-from-code loader function.
    Called by MLflow to load the model from artifacts.

    Args:
        data_path: Path to model artifacts directory containing:
            - config.yaml: Model configuration
            - data/: Document directory with sample images
            - secrets.yaml: Secrets (optional)
            - models/: Model files (base and finetuned models)
            - demo/: Demo folder with UI components (optional)

    Returns:
        Model: Initialized model instance ready for prediction
    """
    from src.mlflow.model import Model

    logger.info(f"Loading Model from artifacts at: {data_path}")

    from src.utils import load_config

    config_path = os.path.join(data_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    config = load_config(config_path)
    logger.info("Configuration loaded successfully")

    # Load secrets if available
    secrets_path = os.path.join(data_path, "secrets.yaml")
    if os.path.exists(secrets_path):
        from src.utils import load_secrets_to_env, load_secrets

        load_secrets_to_env(secrets_path)
        secrets = load_secrets()
        logger.info("Secrets loaded into environment and retrieved")
    else:
        secrets = None

    # Set up models path - for image generation we expect base and finetuned models
    models_path = os.path.join(data_path, "models")
    if not os.path.exists(models_path):
        logger.warning(f"Models directory not found at: {models_path}")
        # Fall back to default model paths from config
        model_no_finetuning_path = config.get(
            "model_no_finetuning_path", "stabilityai/stable-diffusion-xl-base-1.0"
        )
        model_finetuning_path = config.get("model_finetuning_path", "")
    else:
        # Look for base and finetuned models in the models directory
        model_no_finetuning_path = os.path.join(models_path, "model_no_finetuning")
        model_finetuning_path = os.path.join(models_path, "finetuned_model")

        # If they don't exist as directories, fall back to config
        if not os.path.exists(model_no_finetuning_path):
            model_no_finetuning_path = config.get(
                "model_no_finetuning_path", "stabilityai/stable-diffusion-xl-base-1.0"
            )
        if not os.path.exists(model_finetuning_path):
            model_finetuning_path = config.get("model_finetuning_path", "")

    logger.info(f"Base model path: {model_no_finetuning_path}")
    logger.info(f"Finetuned model path: {model_finetuning_path}")

    # Initialize Model
    try:
        model = Model(
            model_no_finetuning_path=model_no_finetuning_path,
            model_finetuning_path=model_finetuning_path,
            config=config,
        )
        logger.info("Model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e
