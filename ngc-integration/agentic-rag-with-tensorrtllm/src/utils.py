"""
Utility functions for agentic RAG with TensorRT-LLM.
"""

import os
from pathlib import Path
from typing import Optional


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
