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
            - fsrcnn_model: PyTorch model file (e.g., FSRCNN_300_epochs.pt)
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
    
    # Get model path from artifacts - the PyTorch model file should be directly in data_path
    # Look for common PyTorch model files
    model_file_candidates = ["FSRCNN_300_epochs.pt", "best_model.pth", "fsrcnn_model"]
    model_path = None
    
    for candidate in model_file_candidates:
        candidate_path = os.path.join(data_path, candidate)
        if os.path.exists(candidate_path):
            model_path = candidate_path
            logger.info(f"Found model file: {model_path}")
            break
    
    if not model_path:
        logger.warning("No PyTorch model file found in artifacts. Model will initialize without pre-trained weights.")
    
    # Set MODEL_ARTIFACTS_PATH for compatibility
    os.environ["MODEL_ARTIFACTS_PATH"] = data_path
    
    # Initialize Model
    try:
        model = Model(
            config=config,
            model_path=model_path
        )
        logger.info("Model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e