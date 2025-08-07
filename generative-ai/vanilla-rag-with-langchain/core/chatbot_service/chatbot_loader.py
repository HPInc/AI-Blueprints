"""
MLflow models-from-code loader module for ChatbotService.
This module provides the _load_pyfunc function required by MLflow's models-from-code approach.
"""

import os
import yaml
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
            - data/: Document directory with AIStudioDoc.pdf
            - secrets.yaml: Encrypted secrets (optional)
            - models/: LLM model files (optional, can be remote path)
            - demo/: Demo folder with UI components (optional)
    
    Returns:
        ChatbotModel: Initialized model instance ready for prediction
    """
    from core.chatbot_service.chatbot_model import ChatbotModel
    
    logger.info(f"Loading ChatbotModel from artifacts at: {data_path}")
    
    # Load configuration
    config_path = os.path.join(data_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully")
    
    # Load secrets if available
    secrets_path = os.path.join(data_path, "secrets.yaml")
    secrets = None
    if os.path.exists(secrets_path):
        with open(secrets_path, 'r') as f:
            secrets = yaml.safe_load(f)
        logger.info("Secrets loaded")
    
    # Set up documents path
    docs_path = os.path.join(data_path, "data")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Documents directory not found at: {docs_path}")
    
    # Set up model path (optional)
    models_path = os.path.join(data_path, "models")
    model_path = None
    if os.path.exists(models_path):
        model_files = [f for f in os.listdir(models_path) if f.endswith(('.gguf', '.bin', '.safetensors'))]
        if model_files:
            model_path = os.path.join(models_path, model_files[0])
            logger.info(f"Local model found at: {model_path}")
    
    # Initialize ChatbotModel
    try:
        chatbot_model = ChatbotModel(
            config=config,
            docs_path=docs_path,
            model_path=model_path,
            secrets=secrets
        )
        logger.info("ChatbotModel initialized successfully")
        return chatbot_model
    except Exception as e:
        logger.error(f"Failed to initialize ChatbotModel: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e
