"""
Utility functions for automated evaluation with structured outputs.

This module contains common functions used across notebooks in the project,
including configuration loading and model initialization.
"""

import os
import yaml
from typing import Dict, Any, Tuple
from pathlib import Path

def load_secrets_to_env(secrets_path: str = "../configs/secrets.yaml") -> None:
    """
    Loads secrets from a YAML file and sets them as environment variables.

    Parameters:
    - secrets_path (str): Path to the secrets YAML file.
    """
    secrets_file = Path(secrets_path).resolve()

    if not secrets_file.exists():
        raise FileNotFoundError(f"Secrets file not found: {secrets_file}")

    with secrets_file.open("r", encoding="utf-8") as file:
        try:
            secrets = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML: {e}")

    if not isinstance(secrets, dict):
        raise ValueError("Secrets file must contain a top-level dictionary.")

    for key, value in secrets.items():
        if not isinstance(key, str):
            raise TypeError(f"Environment variable key must be a string. Got: {type(key)}")
        # We are adding "AIS_" prefix for compatibility with HP AI Studio Secrets Manager.
        os.environ["AIS_" + key] = str(value)

    print(f"✅ Loaded {len(secrets)} secrets into environment variables.")


def load_config(config_path: str = "../configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file is not found.
    """
    # Convert to absolute path if needed
    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml file not found in path: {config_path}")

    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config


def configure_proxy(config: Dict[str, Any]) -> None:
    """
    Configure proxy settings based on provided configuration.

    Args:
        config: Configuration dictionary that may contain a "proxy" key.
    """
    if "proxy" in config and config["proxy"]:
        os.environ["HTTPS_PROXY"] = config["proxy"]
