"""
Code Generation Loader for MLflow models-from-code integration.

This module handles loading configuration and artifacts, then constructs
and returns a configured CodeGenerationModel instance.
"""

import os
import sys
import logging
import yaml
from typing import Dict, Any

from .code_generation_model import CodeGenerationModel
from langchain_huggingface import HuggingFaceEmbeddings
from core.chroma_embedding_adapter import ChromaEmbeddingAdapter
from src.utils import load_secrets_to_env, initialize_llm, get_model_path

# Set up logger
logger = logging.getLogger(__name__)

def _load_pyfunc(data_path: str):
    """
    MLflow models-from-code loader function.
    Called by MLflow to load the code generation model from artifacts.
    
    Args:
        data_path: Path to model artifacts directory containing:
            - config.yaml: Model configuration 
            - models/: LLM model files (optional, can be remote path)
            - embedding_model/: Local embedding model (optional)
    
    Returns:
        CodeGenerationModel: Initialized model instance ready for prediction
    """
    logger.info(f"Loading CodeGenerationModel from artifacts at: {data_path}")
    
    # Load configuration 
    config_path = os.path.join(data_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    config = _load_config(config_path, data_path)
    
    # Initialize embedding function first (critical order)
    embedding_function = _initialize_embedding_function(data_path)
    chroma_embedding_function = ChromaEmbeddingAdapter(embedding_function)
    
    # Initialize LLM
    llm = _initialize_llm(config, data_path)
    
    # Create and return model instance
    model = CodeGenerationModel(
        llm=llm,
        config=config,
        embedding_function=embedding_function,
        chroma_embedding_function=chroma_embedding_function,
        delay_async_init=False  # Enable async components for runtime
    )
    
    logger.info("CodeGenerationModel loaded successfully")
    return model

def _load_config(config_path: str, artifacts_path: str) -> Dict[str, Any]:
    """
    Load configuration from artifacts and environment.
    
    Args:
        config_path: Path to configuration file
        artifacts_path: Base artifacts directory
        
    Returns:
        Configuration dictionary
    """
    # Load secrets into environment if available
    secrets_path = os.path.join(artifacts_path, "secrets")
    if os.path.exists(secrets_path):
        try:
            load_secrets_to_env(secrets_path)
            logger.info(f"Secrets loaded from {secrets_path} into environment")
        except Exception as e:
            logger.warning(f"Failed to load secrets from {secrets_path}: {e}")

    # Retrieve the token from the current environment
    token = os.getenv("AIS_HUGGINGFACE_API_KEY", "")
    if not token.strip():
        logger.warning("Key AIS_HUGGINGFACE_API_KEY not found or empty")
    else:
        logger.info("Hugging Face token is available and loaded from environment")
    
    # Load configuration
    if os.path.exists(config_path):
        with open(config_path) as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
    else:
        config = {}
        logger.warning(f"Configuration file not found at {config_path}")
        
    # Merge configurations
    model_config = {
        "hf_key": token,
        "proxy": config.get("proxy", None),
        "model_source": config.get("model_source", "local"),
    }
    
    # Add model path if local model is specified - check both "models" directory and direct path
    models_dir = os.path.join(artifacts_path, "models")
    if os.path.exists(models_dir):
        # Look for model files in the models directory
        model_files = [f for f in os.listdir(models_dir) if f.endswith(('.gguf', '.bin', '.pt', '.pth'))]
        if model_files:
            model_config["local_model_path"] = os.path.join(models_dir, model_files[0])
            logger.info(f"Found model file in models directory: {model_files[0]}")
    elif "model_path" in config:
        # Use the model path from config if no models directory
        model_config["local_model_path"] = config["model_path"]
        logger.info(f"Using model path from config: {config['model_path']}")
    
    return model_config

def _initialize_embedding_function(artifacts_path: str):
    """
    Initialize the embedding function, checking for artifact model first.
    
    Args:
        artifacts_path: Base artifacts directory
        
    Returns:
        Initialized HuggingFaceEmbeddings object
    """
    logger.info("Initializing embedding function")
    
    # Check for saved embedding model in artifacts
    embedding_model_path = None
    embedding_artifact_path = os.path.join(artifacts_path, "embedding_model")
    if os.path.exists(embedding_artifact_path):
        embedding_model_path = embedding_artifact_path
        logger.info(f"Found saved embedding model in artifacts: {embedding_model_path}")
    
    # Determine which model path to use
    model_name = embedding_model_path if embedding_model_path else "all-MiniLM-L6-v2"
    if embedding_model_path:
        logger.info(f"Using provided embedding model path: {embedding_model_path}")
    else:
        logger.info("Using default embedding model: all-MiniLM-L6-v2")
    
    # Initialize the embedding function
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)
    logger.info(f"Successfully initialized HuggingFaceEmbeddings with model: {model_name}")
        
    return embedding_function

def _initialize_llm(config: Dict[str, Any], artifacts_path: str):
    """
    Initialize the language model based on configuration.
    
    Args:
        config: Configuration dictionary
        artifacts_path: Base artifacts directory
        
    Returns:
        Initialized LLM instance
    """
    model_source = config.get("model_source", "local")
    logger.info(f"Initializing LLM with model source: {model_source}")
    
    # Extract secrets from config
    secrets = {}
    if "hf_key" in config:
        secrets["AIS_HUGGINGFACE_API_KEY"] = config["hf_key"]
    
    # Get local model path from artifacts or config
    local_model_path = None
    if model_source == "local":
        # First check if we have a local_model_path from config processing
        if "local_model_path" in config:
            local_model_path = config["local_model_path"]
            logger.info(f"Using local model path from config: {local_model_path}")
        else:
            # Check for models directory in artifacts
            models_artifact_path = os.path.join(artifacts_path, "models")
            if os.path.exists(models_artifact_path):
                # Look for model files
                model_files = [f for f in os.listdir(models_artifact_path) if f.endswith(('.gguf', '.bin', '.pt', '.pth'))]
                if model_files:
                    local_model_path = os.path.join(models_artifact_path, model_files[0])
                    logger.info(f"Found model file in artifacts: {local_model_path}")
            
            # Use get_model_path utility if no direct path found
            if not local_model_path and "model_path" in config:
                local_model_path = get_model_path(config["model_path"])
                logger.info(f"Resolved model path using utility: {local_model_path}")
    
    # Initialize LLM using the utility function
    llm = initialize_llm(model_source, secrets, local_model_path)
    
    if llm is None:
        logger.error("Failed to initialize LLM from any source")
        raise ValueError("No LLM could be initialized")
        
    logger.info(f"LLM of type {type(llm).__name__} loaded successfully")
    return llm

def save_model(model_config, model_path, sample_input=None, sample_output=None):
    """
    Save model using models-from-code approach.
    
    Args:
        model_config: Configuration for the model
        model_path: Path to save the model
        sample_input: Sample input for signature inference (optional)
        sample_output: Sample output for signature inference (optional)
    """
    import mlflow
    from mlflow.models import infer_signature
    import tempfile
    
    # Infer signature if sample data provided
    signature = None
    if sample_input is not None and sample_output is not None:
        signature = infer_signature(sample_input, sample_output)
    
    # Prepare artifacts
    artifacts = {
        "config": model_config.get("config_path")
    }
    
    # Add optional artifacts
    if model_config.get("secrets_dict"):
        # Save secrets to temporary file
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.safe_dump(model_config["secrets_dict"], tmp)
        tmp.close()
        artifacts["secrets"] = tmp.name
    
    if model_config.get("model_path"):
        artifacts["models"] = model_config["model_path"]
        
    if model_config.get("embedding_model_path"):
        artifacts["embedding_model"] = model_config["embedding_model_path"]
        
    if model_config.get("demo_folder"):
        artifacts["demo"] = model_config["demo_folder"]
    
    # Set up conda environment
    conda_env = {
        "channels": ["defaults", "conda-forge"],
        "dependencies": [
            "python=3.9",
            "pip",
            {
                "pip": [
                    "-r requirements.txt"
                ]
            }
        ],
        "name": "code_generation_env"
    }
    
    # Set up code paths
    code_paths = ["./core", "../src"]
    
    # Save model using models-from-code
    mlflow.models.save_model(
        path=model_path,
        loader_module="code_generation_loader",
        signature=signature,
        conda_env=conda_env,
        code_paths=code_paths
    )
    
    logger.info(f"Model saved to {model_path} using models-from-code approach")
