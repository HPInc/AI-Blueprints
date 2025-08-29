"""
MLflow models-from-code loader module for Logger.
This module provides the _load_pyfunc function required by MLflow's.
"""

import os
import logging
import yaml
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
            - secrets.yaml: Secrets (optional)
            - models/: LLM model files (optional, can be remote path)
            - demo/: Demo folder with UI components (optional)
    
    Returns:
        Model: Initialized model instance ready for prediction
    """
    from src.mlflow.model import Model
    
    logger.info(f"Loading Model from artifacts at: {data_path}")
    
    # Load configuration
    config_path = os.path.join(data_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    config = _load_config(config_path)
    logger.info("Configuration loaded successfully")
    
    # Load secrets if available
    secrets_path = os.path.join(data_path, "secrets.yaml")
    if os.path.exists(secrets_path):
        _load_secrets(secrets_path)
        logger.info("Secrets loaded into environment")
    
    # Initialize Model for text summarization
    try:
        # Import prompt template utilities
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
        from src.prompt_templates import format_summarization_prompt
        
        # Setup environment
        _setup_environment(config)
        
        # Initialize the LLM based on configuration
        llm = _initialize_llm(config, data_path)
        logger.info(f"LLM initialized: {type(llm).__name__}")
        
        # Get formatted prompt template
        model_source = config.get("model_source", "local")
        prompt_str = format_summarization_prompt(model_source)
        logger.info("Prompt template formatted")
        
        # Create Model instance with text summarization specific signature
        model = Model(
            llm=llm,
            config=config,
            prompt_str=prompt_str
        )
        logger.info("Model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                logger.info(f"Configuration loaded from {config_path}")
        else:
            config = {}
            logger.warning(f"Configuration file not found at {config_path}, using empty config")
            
        # Merge with environment-based configuration
        token = os.getenv("AIS_HUGGINGFACE_API_KEY", "")
        if not token.strip():
            logger.warning("Key AIS_HUGGINGFACE_API_KEY not found or empty")
        else:
            logger.info("Hugging Face token is available and loaded from environment")
        
        # Create final configuration
        final_config = {
            "hf_key": token,
            "proxy": config.get("proxy", None),
            "model_source": config.get("model_source", "local"),
        }
        
        return final_config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise


def _load_secrets(secrets_path: str) -> None:
    """Load secrets from file into environment if available."""
    if os.path.exists(secrets_path):
        try:
            from src.utils import load_secrets_to_env
            load_secrets_to_env(secrets_path)
            logger.info(f"Secrets loaded from {secrets_path} into environment")
        except Exception as e:
            logger.warning(f"Failed to load secrets from {secrets_path}: {e}")
    else:
        logger.info("No secrets file found, skipping secret loading")


def _setup_environment(config):
    """Configure environment variables based on loaded configuration."""
    try:
        # Configure proxy if specified in config
        if "proxy" in config and config["proxy"]:
            logger.info(f"Setting up proxy: {config['proxy']}")
            os.environ["HTTPS_PROXY"] = config["proxy"]
            os.environ["HTTP_PROXY"] = config["proxy"]
        else:
            logger.info("No proxy configuration found")
                    
    except Exception as e:
        logger.error(f"Error setting up environment: {str(e)}")


def _initialize_llm(config, data_path):
    """Initialize the LLM based on configuration."""
    from src.mlflow.model import Model
    
    model_source = config.get("model_source", "local")
    logger.info(f"Initializing LLM with source: {model_source}")
    
    if model_source == "local":
        # For text summarization, use model artifacts path
        model_artifacts_path = os.path.join(data_path, "models")
        if os.path.exists(model_artifacts_path):
            model_files = [f for f in os.listdir(model_artifacts_path) if f.endswith(('.gguf', '.bin', '.pt'))]
            if model_files:
                model_path = os.path.join(model_artifacts_path, model_files[0])
                return Model.create_local_llama_model(model_path)
        
        # Fallback to default model path from config
        model_path = config.get("model_path", "")
        if model_path and os.path.exists(model_path):
            return Model.create_local_llama_model(model_path)
        else:
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
    elif model_source == "hugging-face-local":
        hf_token = config.get("hf_key", "")
        return Model.create_local_hf_model(hf_token)
        
    elif model_source == "hugging-face-cloud":
        hf_token = config.get("hf_key", "")
        if not hf_token:
            raise ValueError("Missing required HuggingFace API key for cloud model")
        return Model.create_cloud_hf_model(hf_token)
        
    else:
        raise ValueError(f"Unsupported model source: {model_source}")
