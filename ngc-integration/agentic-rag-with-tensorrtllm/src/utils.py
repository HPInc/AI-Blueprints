"""
Utility functions for agentic RAG with TensorRT-LLM.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any


def get_model_path(model_name: str) -> str:
    """
    Get the full path to the model file using the artifacts path and model name.

    Args:
        model_name: Name of the model file or full path (will extract filename)

    Returns:
        Full path to the model file
    """
    # Extract just the filename if model_name contains a path
    filename = os.path.basename(model_name)

    artifacts_path = os.environ.get("MODEL_ARTIFACTS_PATH", "")
    model_path = os.path.join(artifacts_path, filename)

    return model_path


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_secrets_to_env(secrets_path: str) -> None:
    """
    Load secrets from YAML file and set them as environment variables.

    Args:
        secrets_path: Path to secrets YAML file
    """
    import yaml

    if not os.path.exists(secrets_path):
        return

    with open(secrets_path, "r") as f:
        secrets = yaml.safe_load(f)

    if secrets:
        for key, value in secrets.items():
            os.environ[key] = str(value)


def load_secrets() -> Dict[str, Any]:
    """
    Load secrets from environment variables.

    Returns:
        Dictionary of secrets loaded from environment
    """
    # This can be customized based on what secrets are expected
    # For now, return all environment variables that might be secrets
    secrets = {}
    for key, value in os.environ.items():
        if any(
            secret_key in key.upper()
            for secret_key in ["API_KEY", "TOKEN", "SECRET", "PASSWORD"]
        ):
            secrets[key] = value
    return secrets
