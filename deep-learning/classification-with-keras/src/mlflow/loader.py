"""
MLflow models-from-code loader module for MNIST Classification.
This module provides the _load_pyfunc function required by MLflow's models-from-code approach.
"""

import os
import logging
from typing import Dict, Any, Optional

# Set up logger
logger = logging.getLogger(__name__)


def _load_pyfunc(data_path: str):
    """
    MLflow models-from-code loader function for MNIST classification.
    Called by MLflow to load the model from artifacts.
    
    Args:
        data_path: Path to model artifacts directory containing:
            - config.yaml: Model configuration 
            - model_keras_mnist.keras: Trained Keras model
            - demo/: Demo folder with UI components (optional)
    
    Returns:
        Model: Initialized MNIST model instance ready for prediction
    """
    from src.mlflow.model import Model
    from src.utils import load_config, get_model_path
    
    logger.info(f"Loading MNIST Model from artifacts at: {data_path}")
    
    # Set MODEL_ARTIFACTS_PATH for model loading
    os.environ["MODEL_ARTIFACTS_PATH"] = data_path
    
    config_path = os.path.join(data_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    config = load_config(config_path)
    logger.info("Configuration loaded successfully")
    
    # Get model path from artifacts
    model_path = get_model_path("model_keras_mnist.keras")
    
    if not os.path.exists(model_path):
        # Fallback to direct path in artifacts
        model_path = os.path.join(data_path, "model_keras_mnist.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    logger.info(f"Model path resolved to: {model_path}")
    
    # Initialize and return the Model
    model = Model(
        config=config,
        model_path=model_path
    )
    
    logger.info("MNIST Model initialized successfully")
    return model
    try:
        model = Model(
            config=config,
            docs_path=docs_path,
            secrets=secrets,
            model_path=model_path
        )
        logger.info("Model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e