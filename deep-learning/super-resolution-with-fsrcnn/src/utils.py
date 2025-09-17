"""
Utility functions for AI Studio Deep Learning Templates.

This module contains common functions used across notebooks in the project,
including configuration loading, model initialization, and other utilities.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


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


def get_ui_mode(config: Dict[str, Any]) -> str:
    """
    Get the UI mode from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        UI mode string (static, streamlit, or gradio).
    """
    return config.get("ui", {}).get("mode", "static")


def get_service_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get service configuration settings.

    Args:
        config: Configuration dictionary.

    Returns:
        Service configuration dictionary.
    """
    return config.get("service", {})


def get_ports_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get port configuration settings.

    Args:
        config: Configuration dictionary.

    Returns:
        Ports configuration dictionary.
    """
    return config.get("ports", {})
