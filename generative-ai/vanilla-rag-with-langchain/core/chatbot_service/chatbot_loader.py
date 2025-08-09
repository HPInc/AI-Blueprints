"""
MLflow models-from-code loader module for ChatbotService.
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
            - data/: Document directory with AIStudioDoc.pdf
            - secrets.yaml: Secrets (optional)
            - models/: LLM model files (optional, can be remote path)
            - demo/: Demo folder with UI components (optional)
    
    Returns:
        ChatbotModel: Initialized model instance ready for prediction
    """
    from core.chatbot_service.chatbot_model import ChatbotModel
    
    logger.info(f"Loading ChatbotModel from artifacts at: {data_path}")
    
    from src.utils import load_config, load_secrets_to_env, load_secrets
    
    config_path = os.path.join(data_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    config = load_config(config_path)
    logger.info("Configuration loaded successfully")
    
    # Load secrets if available
    secrets_path = os.path.join(data_path, "secrets.yaml")
    if os.path.exists(secrets_path):
        load_secrets_to_env(secrets_path)
        secrets = load_secrets()
        logger.info("Secrets loaded into environment and retrieved")
    else:
        secrets = None
    
    # Set up documents path
    docs_path = os.path.join(data_path, "data")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Documents directory not found at: {docs_path}")
    
    # Get model path from config (fallback to None if not specified - ChatbotModel handles defaults)
    model_path = config.get("model_path")
    if model_path:
        logger.info(f"Using model path from config: {model_path}")
    else:
        logger.info("No model_path found in config, ChatbotModel will use default fallback")
    
    # Initialize ChatbotModel
    try:
        chatbot_model = ChatbotModel(
            config=config,
            docs_path=docs_path,
            secrets=secrets,
            model_path=model_path
        )
        logger.info("ChatbotModel initialized successfully")
        return chatbot_model
    except Exception as e:
        logger.error(f"Failed to initialize ChatbotModel: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e
