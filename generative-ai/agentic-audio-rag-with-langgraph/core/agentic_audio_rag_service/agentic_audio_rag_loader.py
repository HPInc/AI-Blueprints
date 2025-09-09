"""
MLflow models-from-code loader module for AgenticAudioService.
This module provides the _load_pyfunc function required by MLflow.
"""
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from core.agentic_audio_rag_service.agentic_audio_rag_model import AgenticAudioModel
from src.utils import load_config, load_secrets_to_env, load_secrets


# Set up logger
logger = logging.getLogger(__name__)

class _Context:
        def __init__(self, artifacts: Dict[str, str]):
            self.artifacts = artifacts

def _load_pyfunc(data_path: str):
    """
    MLflow models-from-code loader function.
    Called by MLflow to load the model from artifacts.
    
    Args:
        data_path: Path to model artifacts directory containing:
            - config.yaml: Model configuration 
            - data/input: File directory with mp3, wav, mp4, etc.
            - secrets.yaml: Secrets (optional)
            - models/: LLM model files (optional, can be remote path)
            - demo/: Demo folder with UI components (optional)
    
    Returns:
        AgenticAudioModel: Initialized model instance ready for prediction
    """
    logger.info(f"Loading AgenticAudioModel from artifacts at: {data_path}")
        
    root = Path(data_path)
    
    (root / "index").mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "memory").mkdir(parents=True, exist_ok=True)  
    
    artifacts = {
        "index_dir": str(root / "index"),
        "config_path": str(root / "config" / "config.json"),
        "memory_dir": str(root / "memory"),
    }
    ctx = _Context(artifacts)
    
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
        raise FileNotFoundError(f"File directory not found at: {docs_path}")
    
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
        logger.info("No model_path found in config, AgenticAudioModel will use default fallback")
    
    # Initialize AgenticAudioModel
    try:
        agentic_audio_model = AgenticAudioModel(
            context=ctx,
            config=config,
            docs_path=docs_path,
            secrets=secrets,
            model_path=model_path
        )
        logger.info("AgenticAudioModel initialized successfully")
        return agentic_audio_model
    except Exception as e:
        logger.error(f"Failed to initialize AgenticAudioModel: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e
