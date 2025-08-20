"""
MLflow models-from-code loader module for EvaluationService.
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
    Called by MLflow to load the evaluation model from artifacts.
    
    Args:
        data_path: Path to model artifacts directory containing:
            - config.yaml: Model configuration 
            - models/: LLM model files (optional, can be remote path)
    
    Returns:
        EvaluationModel: Initialized model instance ready for prediction
    """
    from core.evaluation_service.evaluation_model import EvaluationModel
    
    logger.info(f"Loading EvaluationModel from artifacts at: {data_path}")
    
    from src.utils import load_config
    
    config_path = os.path.join(data_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    config = load_config(config_path)
    logger.info("Configuration loaded successfully")
    
    # Get model path from config and resolve it for MLflow artifacts context
    model_path = config.get("model_path")
    if model_path:
        from src.utils import get_model_path
        
        # Set MODEL_ARTIFACTS_PATH for get_model_path function
        os.environ["MODEL_ARTIFACTS_PATH"] = data_path
        
        # Resolve model path relative to artifacts
        resolved_model_path = get_model_path(model_path)
        model_path = resolved_model_path
    else:
        logger.info("No model_path found in config, EvaluationModel will use default fallback")
    
    # Initialize EvaluationModel
    try:
        evaluation_model = EvaluationModel(
            llm_model_path=model_path,
            config=config
        )
        logger.info("EvaluationModel initialized successfully")
        return evaluation_model
    except Exception as e:
        logger.error(f"Failed to initialize EvaluationModel: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e