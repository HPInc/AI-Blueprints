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
    Called by MLflow to load the model from artifacts.
    
    Args:
        data_path: Path to model artifacts directory containing:
            - config.yaml: Model configuration 
            - llama_model_path: Path to LLaMA model file
            - other artifacts
    
    Returns:
        EvaluationModel: Initialized model instance ready for prediction
    """
    from core.evaluation_service.evaluation_model import EvaluationModel
    
    logger.info(f"Loading EvaluationModel from artifacts at: {data_path}")
    
    # Get model path from environment variable (set by MLflow artifacts context)
    model_path = os.environ.get("MODEL_ARTIFACTS_PATH", data_path)
    
    # Load configuration if available
    config_path = os.path.join(data_path, "config.yaml")
    config = {}
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
    
    # Determine LLM model path - this should be set via artifacts
    # The actual model file will be available through MLflow artifacts
    llm_model_path = None
    
    # Check for model file in artifacts
    potential_model_files = [
        "llama_model_path",  # Direct artifact reference
        "model.gguf",        # Common GGUF file name
        "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"  # Specific model name
    ]
    
    for model_file in potential_model_files:
        potential_path = os.path.join(data_path, model_file)
        if os.path.exists(potential_path):
            llm_model_path = potential_path
            break
    
    # If no model found in artifacts, try to use MODEL_ARTIFACTS_PATH
    if not llm_model_path:
        artifacts_path = os.environ.get("MODEL_ARTIFACTS_PATH", "")
        if artifacts_path:
            for model_file in potential_model_files:
                potential_path = os.path.join(artifacts_path, model_file)
                if os.path.exists(potential_path):
                    llm_model_path = potential_path
                    break
    
    if not llm_model_path:
        raise FileNotFoundError(f"LLaMA model file not found in artifacts at: {data_path}")
    
    # Initialize EvaluationModel
    try:
        evaluation_model = EvaluationModel(
            llm_model_path=llm_model_path,
            config=config
        )
        logger.info("EvaluationModel initialized successfully")
        return evaluation_model
    except Exception as e:
        logger.error(f"Failed to initialize EvaluationModel: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e
