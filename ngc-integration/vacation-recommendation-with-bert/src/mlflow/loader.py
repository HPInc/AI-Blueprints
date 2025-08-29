"""
MLflow models-from-code loader module for BERT Tourism Recommendation.
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
    Called by MLflow to load the BERT tourism model from artifacts.

    Args:
        data_path: Path to model artifacts directory containing:
            - config.yaml: Model configuration
            - embeddings.csv: Precomputed embeddings
            - corpus.csv: Tourism corpus data
            - tokenizer/: BERT tokenizer directory
            - models/: BERT model files
            - demo/: Demo folder (optional)

    Returns:
        Model: Initialized BERT tourism model ready for vacation recommendations
    """
    from src.mlflow.model import Model

    logger.info(f"Loading BERT Tourism Model from artifacts at: {data_path}")

    # Load configuration
    config_path = os.path.join(data_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    # For now, we'll use a simple config loading approach since utils.py doesn't have load_config yet
    import yaml
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise

    # Set MODEL_ARTIFACTS_PATH environment variable for artifact context
    os.environ["MODEL_ARTIFACTS_PATH"] = data_path

    # Resolve artifact paths
    embeddings_path = os.path.join(data_path, "embeddings.csv")
    corpus_path = os.path.join(data_path, "corpus.csv")
    tokenizer_dir = os.path.join(data_path, "tokenizer")
    
    # BERT model path - check if it exists in artifacts or use config fallback
    bert_model_artifact_path = os.path.join(data_path, "bert_model.nemo")
    if os.path.exists(bert_model_artifact_path):
        bert_model_path = bert_model_artifact_path
        logger.info(f"Using BERT model from artifacts: {bert_model_path}")
    else:
        # Fallback to paths from original configuration
        bert_model_path = "/home/jovyan/datafabric/Bertlargeuncased/bertlargeuncased.nemo"
        logger.info(f"Using fallback BERT model path: {bert_model_path}")

    # Validate required artifact files
    required_files = {
        "embeddings": embeddings_path,
        "corpus": corpus_path,
        "tokenizer": tokenizer_dir,
    }
    
    for name, path in required_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name.capitalize()} not found at: {path}")

    # Initialize BERT Tourism Model
    try:
        model = Model(
            embeddings_path=embeddings_path,
            corpus_path=corpus_path,
            tokenizer_dir=tokenizer_dir,
            bert_model_path=bert_model_path,
            config=config
        )
        logger.info("BERT Tourism Model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize BERT Tourism Model: {str(e)}")
        raise RuntimeError(f"BERT Tourism Model loading failed: {str(e)}") from e