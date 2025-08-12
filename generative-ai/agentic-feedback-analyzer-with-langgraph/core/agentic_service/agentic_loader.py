"""
MLflow models-from-code loader module for AgenticFeedbackService.
This module provides the _load_pyfunc function required by MLflow.
"""

import os
import logging

logger = logging.getLogger(__name__)


def _load_pyfunc(data_path: str):
    """
    MLflow models-from-code loader function.
    Called by MLflow to load the model from artifacts.
    
    Args:
        data_path: Path to model artifacts directory containing:
            - config.yaml: Model configuration 
            - data/: Document directory 
            - memory/: Memory storage directory
            - models/: LLM model files (optional)
            - demo/: Demo folder (optional)
    
    Returns:
        AgenticModel: Initialized model instance ready for prediction
    """
    from core.agentic_service.agentic_model import AgenticModel
    import sys
    import os
    
    # Add src to path for utils import
    src_path = os.path.join(os.path.dirname(data_path), "src")
    if src_path not in sys.path:
        sys.path.append(src_path)
    
    from src.utils import load_config
    
    logger.info(f"Loading AgenticModel from artifacts at: {data_path}")
    
    # Load configuration using helper function
    config_path = os.path.join(data_path, "config.yaml")
    config = load_config(config_path)
    if not config:
        raise FileNotFoundError(f"Configuration file not found or invalid at: {config_path}")
    
    # Set up paths
    docs_path = os.path.join(data_path, "data")
    memory_path = os.path.join(data_path, "memory")
    
    # Get model path from config and resolve it for MLflow artifacts context
    model_path = config.get("model_path")
    if model_path:
        # Set MODEL_ARTIFACTS_PATH for artifact resolution
        os.environ["MODEL_ARTIFACTS_PATH"] = data_path
        
        # Check if model file exists in models subdirectory
        models_dir = os.path.join(data_path, "models")
        if os.path.exists(models_dir):
            model_filename = os.path.basename(model_path)
            artifact_model_path = os.path.join(models_dir, model_filename)
            if os.path.exists(artifact_model_path):
                model_path = artifact_model_path
                logger.info(f"Using model from artifacts: {model_path}")
    else:
        logger.info("No model_path found in config")
    
    # Initialize AgenticModel
    try:
        agentic_model = AgenticModel(
            config=config,
            docs_path=docs_path,
            memory_path=memory_path,
            model_path=model_path
        )
        logger.info("AgenticModel initialized successfully")
        return agentic_model
    except Exception as e:
        logger.error(f"Failed to initialize AgenticModel: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e

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
            - config.yaml: Model configuration (optional)
            - model_path: LLM model file path (from artifacts)
            - memory_path: Memory storage directory (from artifacts)
    
    Returns:
        AgenticModel: Initialized model instance ready for prediction
    """
    from core.agentic_service.agentic_model import AgenticModel
    
    logger.info(f"Loading AgenticModel from artifacts at: {data_path}")
    
    # Get model path from artifacts - model should be placed in artifacts during registration
    model_path = os.path.join(data_path, "model_path")
    if not os.path.exists(model_path):
        # Fallback to environment variable or default path
        model_path = os.environ.get("MODEL_PATH", "/home/jovyan/datafabric/meta-llama3.1-8b-Q8/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
        logger.warning(f"Model path not found in artifacts, using fallback: {model_path}")
    
    # Get memory path from artifacts or use default
    memory_path = os.path.join(data_path, "memory_path")
    if not os.path.exists(memory_path):
        # Create memory directory in artifacts if it doesn't exist
        memory_path = os.path.join(data_path, "memory")
        os.makedirs(memory_path, exist_ok=True)
        logger.info(f"Created memory directory at: {memory_path}")
    
    # Validate model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    # Initialize AgenticModel
    try:
        agentic_model = AgenticModel(
            model_path=model_path,
            memory_path=memory_path
        )
        logger.info("AgenticModel initialized successfully")
        return agentic_model
    except Exception as e:
        logger.error(f"Failed to initialize AgenticModel: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e


def save_model(model_path: str, memory_path: str, artifact_path: str, sample_input=None, sample_output=None):
    """
    Save model using models-from-code approach.
    
    Args:
        model_path: Path to the LLM model file
        memory_path: Path to memory storage directory
        artifact_path: MLflow artifact path for registration
        sample_input: Sample input for signature inference
        sample_output: Sample output for signature inference
    """
    import mlflow
    from mlflow.models import infer_signature
    import os
    import shutil
    
    # Determine data path for artifacts
    data_path = "agentic_model_data"
    
    # Prepare artifacts dictionary
    artifacts = {}
    
    # Copy model file to artifacts if it exists locally
    if os.path.exists(model_path):
        artifacts["model_path"] = model_path
    
    # Copy memory directory to artifacts
    if os.path.exists(memory_path):
        artifacts["memory_path"] = memory_path
    
    # Infer signature if sample data provided
    signature = None
    if sample_input is not None and sample_output is not None:
        signature = infer_signature(sample_input, sample_output)
    
    # Define conda environment
    conda_env = {
        "channels": ["conda-forge", "defaults"],
        "dependencies": [
            "python=3.11",
            "pip",
            {
                "pip": [
                    "mlflow==3.1.0",
                    "langchain-community",
                    "langgraph", 
                    "pydantic",
                    "pandas",
                    "llama-cpp-python",
                ]
            }
        ],
        "name": "agentic_env"
    }
    
    # Define code paths to include
    code_paths = ["../src"]
    
    # Save model using models-from-code
    mlflow.models.save_model(
        path=artifact_path,
        loader_module="agentic_loader",
        data_path=data_path,
        artifacts=artifacts,
        signature=signature,
        conda_env=conda_env,
        code_paths=code_paths
    )
    
    logger.info(f"Model saved successfully to: {artifact_path}")
