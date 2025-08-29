"""
MLflow models-from-code loader module for Logger.
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
            - models/: LLM model files (*.gguf)
            - secrets.yaml: Secrets (optional)
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
    
    # Load secrets if available
    secrets_path = os.path.join(data_path, "secrets.yaml")
    if os.path.exists(secrets_path):
        from src.utils import load_secrets_to_env, load_secrets
        load_secrets_to_env(secrets_path)
        secrets = load_secrets()
        logger.info("Secrets loaded into environment and retrieved")
    else:
        secrets = None
    
    # Load LLM from models directory
    models_path = os.path.join(data_path, "models")
    llm = _load_llm_from_artifacts(models_path, config)
    
    # Initialize Model (text-generation specific)
    try:
        model = Model(llm=llm, config=config)
        logger.info("Model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e


def _load_llm_from_artifacts(models_path: str, config: Dict[str, Any]):
    """Load the LlamaCpp model from artifacts."""
    from src.utils import configure_hf_cache, configure_proxy
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain_community.llms import LlamaCpp
    import time
    import glob

    if hasattr(LlamaCpp, "model_rebuild"):
        LlamaCpp.model_rebuild()

    # Find *.gguf model file in models directory
    model_files = glob.glob(os.path.join(models_path, "*.gguf"))
    if not model_files:
        raise RuntimeError(f"No *.gguf model file found in {models_path}")
    
    model_path = model_files[0]  # Use first found model
    logger.info(f"Using model file: {model_path}")

    configure_hf_cache()
    configure_proxy(config)

    start = time.perf_counter()
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=int(config.get("n_gpu_layers", 1)),  # 0 → CPU-only
        n_batch=256,
        n_ctx=4096,
        max_tokens=1024,
        f16_kv=True,
        temperature=0.2,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
        streaming=False,
    )
    logger.info("🔹 LlamaCpp loaded in %.1fs", time.perf_counter() - start)
    return llm
