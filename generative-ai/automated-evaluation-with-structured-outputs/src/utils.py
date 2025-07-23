"""
Utility functions for automated evaluation with structured outputs.

This module contains common functions used across notebooks in the project,
including configuration loading and model initialization.
"""

import os
import yaml
from typing import Dict, Any, Tuple


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
