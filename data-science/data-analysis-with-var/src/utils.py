"""
Utility functions for the data-analysis-with-var blueprint.
"""

import os
import yaml
from typing import Dict, Any


def load_config(
    config_path: str = "../../configs/config.yaml"
) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration YAML file.
        secrets_path: Path to the secrets YAML file.

    Returns:
        Dictionary containing the project configurations.

    Raises:
        FileNotFoundError: If the config file is not found.
    """
    # Convert to absolute paths if needed
    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml file not found in path: {config_path}")

    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config


def get_model_path(model_name: str) -> str:
    """
    Get the full path to the model file using the artifacts path and model name.
    This is a compatibility function for the generic loader.

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


def load_secrets_to_env(secrets_path: str) -> None:
    """
    Load secrets from YAML file to environment variables.
    This is a compatibility function for the generic loader.

    Args:
        secrets_path: Path to the secrets YAML file
    """
    if not os.path.exists(secrets_path):
        return
    
    with open(secrets_path) as file:
        secrets = yaml.safe_load(file)
        if secrets:
            for key, value in secrets.items():
                os.environ[key] = str(value)


def load_secrets() -> Dict[str, Any]:
    """
    Load secrets from environment variables.
    This is a compatibility function for the generic loader.

    Returns:
        Dictionary containing secrets from environment
    """
    # For data-analysis-with-var, we typically don't use secrets
    # This is just for compatibility with the generic loader
    return {}