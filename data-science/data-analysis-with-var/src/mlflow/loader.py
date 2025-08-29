"""
MLflow models-from-code loader module for COVID Movement Patterns with VAR.
This module provides the _load_pyfunc function required by MLflow's models-from-code.
"""

import os
import pickle
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
            - ny_model.pkl: New York VAR model
            - ldn_model.pkl: London VAR model
            - ny_last_values.pkl: New York last values for forecasting
            - ldn_last_values.pkl: London last values for forecasting
            - ny_last_raw_value.pkl: New York last raw values
            - ldn_last_raw_value.pkl: London last raw values
            - features.pkl: Feature names
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
    
    # Load all the model artifacts
    artifacts = {}
    artifact_files = [
        "ny_model.pkl",
        "ldn_model.pkl", 
        "ny_last_values.pkl",
        "ldn_last_values.pkl",
        "ny_last_raw_value.pkl",
        "ldn_last_raw_value.pkl",
        "features.pkl"
    ]
    
    for artifact_file in artifact_files:
        artifact_path = os.path.join(data_path, artifact_file)
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Required artifact not found: {artifact_path}")
        
        with open(artifact_path, "rb") as f:
            artifact_name = artifact_file.replace(".pkl", "")
            artifacts[artifact_name] = pickle.load(f)
        
        logger.info(f"Loaded artifact: {artifact_file}")
    
    # Initialize Model
    try:
        model = Model(config=config, **artifacts)
        logger.info("Model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e