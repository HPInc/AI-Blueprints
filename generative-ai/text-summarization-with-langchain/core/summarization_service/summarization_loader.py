"""
MLflow models-from-code loader module for SummarizationService.
This module provides the _load_pyfunc function required by MLflow's models-from-code approach.
"""

import os
import logging
import yaml
from typing import Dict, Any, Optional
from .summarization_model import SummarizationModel

# Set up logger
logger = logging.getLogger(__name__)

# Import prompt template utilities
import sys
import os
# Add the src directory to the path
current_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_dir, "../../src"))
sys.path.append(src_path)

from utils import load_secrets_to_env
from prompt_templates import format_summarization_prompt


def _load_pyfunc(data_path: str):
    """
    MLflow models-from-code loader function.
    Called by MLflow to load the model from artifacts.
    
    Args:
        data_path: Path to model artifacts directory containing:
            - config/config.yaml: Model configuration
            - secrets/secrets.yaml: API keys and secrets (optional)
            - model/: Model files directory (for local models)
            
    Returns:
        SummarizationModel: Configured model instance ready for inference
    """
    try:
        logger.info(f"Loading summarization model from data path: {data_path}")
        
        # Load configuration
        config_path = os.path.join(data_path, "config.yaml")
        config = _load_config(config_path)
        logger.info("Configuration loaded successfully")
        
        # Load secrets if available
        secrets_path = os.path.join(data_path, "secrets.yaml")
        _load_secrets(secrets_path)
        
        # Setup environment
        _setup_environment(config)
        
        # Initialize the LLM based on configuration
        llm = _initialize_llm(config, data_path)
        logger.info(f"LLM initialized: {type(llm).__name__}")
        
        # Get formatted prompt template
        model_source = config.get("model_source", "local")
        prompt_str = format_summarization_prompt(model_source)
        logger.info("Prompt template formatted")
        
        # Create and return the SummarizationModel instance
        model = SummarizationModel(
            llm=llm,
            config=config,
            prompt_str=prompt_str
        )
        
        logger.info("SummarizationModel loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def _load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration settings
    """
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
    """
    Load secrets from file into environment if available.
    
    Args:
        secrets_path: Path to secrets file
    """
    if os.path.exists(secrets_path):
        try:
            load_secrets_to_env(secrets_path)
            logger.info(f"Secrets loaded from {secrets_path} into environment")
        except Exception as e:
            logger.warning(f"Failed to load secrets from {secrets_path}: {e}")
    else:
        logger.info("No secrets file found, skipping secret loading")


def _setup_environment(config: Dict[str, Any]) -> None:
    """
    Configure environment variables based on loaded configuration.
    
    Args:
        config: Configuration dictionary
    """
    try:
        # Configure proxy if specified in config
        if "proxy" in config and config["proxy"]:
            logger.info(f"Setting up proxy: {config['proxy']}")
            os.environ["HTTPS_PROXY"] = config["proxy"]
            os.environ["HTTP_PROXY"] = config["proxy"]
        else:
            logger.info("No proxy configuration found. Checking system environment variables.")
            # Check if proxy is set in environment variables
            system_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
            if system_proxy:
                logger.info(f"Using system proxy: {system_proxy}")
            else:
                logger.warning("No proxy configuration found in config or environment variables.")
                
    except Exception as e:
        logger.error(f"Error setting up environment: {str(e)}")
        # Continue without failing to allow the model to still function


def _initialize_llm(config: Dict[str, Any], data_path: str):
    """
    Initialize the LLM based on configuration.
    
    Args:
        config: Configuration dictionary
        data_path: Path to model artifacts
        
    Returns:
        Initialized LLM instance
    """
    model_source = config.get("model_source", "local")
    logger.info(f"Initializing LLM with source: {model_source}")
    
    if model_source == "local":
        model_path = os.path.join(data_path, "model")
        # Find the actual model file in the model directory
        if os.path.isdir(model_path):
            model_files = [f for f in os.listdir(model_path) if f.endswith(('.gguf', '.bin', '.pt'))]
            if model_files:
                model_path = os.path.join(model_path, model_files[0])
            else:
                # If no specific model files, assume model_path is correct
                pass
        return SummarizationModel.create_local_llama_model(model_path)
        
    elif model_source == "hugging-face-local":
        hf_token = config.get("hf_key", "")
        return SummarizationModel.create_local_hf_model(hf_token)
        
    elif model_source == "hugging-face-cloud":
        hf_token = config.get("hf_key", "")
        if not hf_token:
            raise ValueError("Missing required HuggingFace API key for cloud model")
        return SummarizationModel.create_cloud_hf_model(hf_token)
        
    else:
        raise ValueError(f"Unsupported model source: {model_source}")


def save_model(model_config: Dict[str, Any], model_path: str, sample_input=None, sample_output=None):
    """
    Save model using models-from-code approach.
    
    Args:
        model_config: Configuration for the model
        model_path: Path where to save the model
        sample_input: Sample input for signature inference (optional)
        sample_output: Sample output for signature inference (optional)
    """
    import mlflow
    from mlflow.models import infer_signature
    
    # Infer signature if sample data provided
    signature = None
    if sample_input is not None and sample_output is not None:
        signature = infer_signature(sample_input, sample_output)
    
    # Prepare conda environment
    conda_env = {
        'channels': ['defaults', 'conda-forge'],
        'dependencies': [
            'python=3.10',
            'pip',
            {
                'pip': [
                    'mlflow',
                    'langchain',
                    'langchain-community',
                    'langchain-core',
                    'langchain-huggingface',
                    'transformers',
                    'torch',
                    'pandas',
                    'PyYAML'
                ]
            }
        ],
        'name': 'summarization_env'
    }
    
    # Code paths to include
    code_paths = ["core", "src"]
    
    # Save model using models-from-code approach
    mlflow.models.save_model(
        path=model_path,
        loader_module="summarization_loader",
        data_path=None,  # Will be set during logging
        signature=signature,
        conda_env=conda_env,
        code_paths=code_paths
    )
