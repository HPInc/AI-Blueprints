"""
MLflow models-from-code loader module for Logger.
This module provides the _load_pyfunc function required by MLflow's.
"""

import os
import logging
from typing import Dict, Any, Optional

# Set up logger
logger = logging.getLogger(__name__)


def _load_pyfunc(data_path: str):
    """
    MLflow models-from-code loader function.
    Called by MLflow to load the model from artifacts.
    
    Args:
        data_path: Path to model artifacts directory containing:
            - config.yaml: Model configuration 
            - model/: NeMo model files (.nemo files)
            - demo/: Demo folder with UI components (optional)
    
    Returns:
        Model: Initialized model instance ready for prediction
    """
    from src.mlflow.model import Model
    
    logger.info(f"Loading Model from artifacts at: {data_path}")
    
    from src.utils import load_config
    
    config_path = os.path.join(data_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    config = load_config(config_path)
    logger.info("Configuration loaded successfully")
    
    # Set up model directory path for NeMo models
    model_dir = os.path.join(data_path, "model")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found at: {model_dir}")
    
    # NeMo models dictionary - empty since we'll load from model_dir
    nemo_models = {}
    
    # Initialize Model
    try:
        model = Model(
            config=config,
            nemo_models=nemo_models,
            model_dir=model_dir
        )
        logger.info("Model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e