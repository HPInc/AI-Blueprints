"""
MLflow models-from-code loader module.
This module provides the _load_pyfunc function required by MLflow's models-from-code approach.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Set up logger
logger = logging.getLogger(__name__)


def _load_pyfunc(data_path: str):
    """
    MLflow models-from-code loader function.
    Called by MLflow to load the model from artifacts.

    Args:
        data_path: Path to model artifacts directory containing:
            - config.yaml: Model configuration
            - model_no_finetuning/: Base model directory or HF ID file
            - finetuned_model/: Fine-tuned model directory or HF ID file
            - secrets.yaml: Secrets (optional)
            - demo/: Demo folder with UI components (optional)

    Returns:
        Model: Initialized model instance ready for prediction
    """
    from src.mlflow.model import Model

    logger.info(f"Loading Model from artifacts at: {data_path}")

    # Helper function to load config
    def load_config(config_path: str) -> dict:
        """Load configuration from YAML file."""
        import yaml

        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    # Helper function to load secrets if available
    def load_secrets_if_available(secrets_path: str) -> Optional[dict]:
        """Load secrets if file exists."""
        if os.path.exists(secrets_path):
            import yaml

            with open(secrets_path, "r") as file:
                secrets = yaml.safe_load(file)

            # Load secrets into environment
            for key, value in secrets.items():
                os.environ[key] = str(value)

            logger.info("Secrets loaded into environment")
            return secrets
        return None

    # Load configuration
    config_path = os.path.join(data_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    config = load_config(config_path)
    logger.info("Configuration loaded successfully")

    # Load secrets if available
    secrets_path = os.path.join(data_path, "secrets.yaml")
    secrets = load_secrets_if_available(secrets_path)

    # Resolve model paths
    base_model_artifact = os.path.join(data_path, "model_no_finetuning")
    finetuned_model_artifact = os.path.join(data_path, "finetuned_model")

    def resolve_model_path(artifact_path: str) -> str:
        """Resolve model path from artifact, handling both directories and HF ID files."""
        if os.path.isdir(artifact_path):
            # It's a directory containing the model
            return artifact_path
        elif os.path.isfile(artifact_path):
            # It might be a file containing a HuggingFace model ID
            try:
                with open(artifact_path, "r") as f:
                    content = f.read().strip()
                if content:
                    logger.info(f"Using HuggingFace model ID from file: {content}")
                    return content
            except Exception as e:
                logger.warning(
                    f"Failed to read model ID from file {artifact_path}: {e}"
                )

        # Fallback: return the path as-is (might be a HF model ID)
        return artifact_path

    base_model_path = resolve_model_path(base_model_artifact)
    finetuned_model_path = resolve_model_path(finetuned_model_artifact)

    logger.info(f"Resolved base model path: {base_model_path}")
    logger.info(f"Resolved fine-tuned model path: {finetuned_model_path}")

    # Initialize Model
    try:
        model = Model(
            config=config,
            base_model_path=base_model_path,
            finetuned_model_path=finetuned_model_path,
        )
        logger.info("Model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e
