"""
MLflow models-from-code loader module for Spam Detection.
This module provides the _load_pyfunc function required by MLflow's models-from-code approach.
"""

import os
import logging
from typing import Dict, Any, Optional

# Set up logger
logger = logging.getLogger(__name__)


def _load_pyfunc(data_path: str):
    """
    MLflow models-from-code loader function.
    Called by MLflow to load the spam detection model from artifacts.
    
    Args:
        data_path: Path to model artifacts directory containing:
            - config.yaml: Model configuration 
            - spam_utf8.csv: Spam dataset
            - nltk_data/: NLTK data directory
            - demo/: Demo folder with UI components (optional)
    
    Returns:
        Model: Initialized spam detection model instance ready for prediction
    """
    from src.mlflow.model import Model
    
    logger.info(f"Loading Spam Detection Model from artifacts at: {data_path}")
    
    from src.utils import load_config
    
    config_path = os.path.join(data_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    config = load_config(config_path)
    logger.info("Configuration loaded successfully")
    
    # Set up data path for spam dataset
    data_file_path = os.path.join(data_path, "spam_utf8.csv")
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Spam dataset not found at: {data_file_path}")
    
    # Set up NLTK data path
    nltk_data_path = os.path.join(data_path, "nltk_data")
    if not os.path.exists(nltk_data_path):
        raise FileNotFoundError(f"NLTK data directory not found at: {nltk_data_path}")
    
    # Initialize Model
    try:
        model = Model(
            data_path=data_file_path,
            nltk_data_path=nltk_data_path,
            config=config
        )
        logger.info("Spam Detection Model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Spam Detection Model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e