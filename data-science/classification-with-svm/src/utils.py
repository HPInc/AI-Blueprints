"""
Utility functions for Iris Classification with SVM.
This module provides configuration loading utilities.
"""
import os
import yaml
import logging
from typing import Dict, Any, Optional

# Set up logger
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the loaded configuration
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded successfully from: {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {str(e)}")
        raise