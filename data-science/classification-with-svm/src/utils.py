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