"""
MLflow models-from-code loader module for Logger.
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
            - data/: Document directory with AIStudioDoc.pdf
            - secrets.yaml: Secrets (optional)
            - models/: LLM model files (optional, can be remote path)
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
        logger.info("No secrets file found, proceeding without secrets")

    # Get model path from config and resolve it for MLflow artifacts context
    model_path = config.get("model_path")
    if model_path:
        from src.utils import get_model_path

        # Set MODEL_ARTIFACTS_PATH for get_model_path function
        # In the artifacts structure, models are stored in the models/ subdirectory
        models_artifacts_path = os.path.join(data_path, "models")
        os.environ["MODEL_ARTIFACTS_PATH"] = models_artifacts_path

        # Resolve model path relative to artifacts
        resolved_model_path = get_model_path(model_path)
        model_path = resolved_model_path
        logger.info(f"Resolved model path: {model_path}")
    else:
        logger.info("No model_path found in config, Model will use default fallback")

    # Initialize Model
    try:
        # Create context object with artifacts paths
        from types import SimpleNamespace

        context = SimpleNamespace(
            artifacts={
                "config_path": config_path,
                "memory_dir": os.path.join(data_path, "memory"),
                "index_dir": os.path.join(
                    data_path, "indexes"
                ),  # Changed from "index" to "indexes"
                "models": os.path.join(data_path, "models"),  # Add models directory
            }
        )

        model = Model(
            context=context,
            config=config,
            model_path=model_path,
            secrets=secrets,
        )
        logger.info("Model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e
