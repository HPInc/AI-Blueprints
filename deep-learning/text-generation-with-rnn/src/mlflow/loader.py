"""
MLflow models-from-code loader module for RNN Text Generation.
This module provides the _load_pyfunc function required by MLflow's models-from-code approach.
"""

import os
import sys
import logging
import torch
from typing import Dict, Any, Optional

# Set up logger
logger = logging.getLogger(__name__)


def _load_pyfunc(data_path: str):
    """
    MLflow models-from-code loader function.
    Called by MLflow to load the RNN text generation model from artifacts.

    Args:
        data_path: Path to model artifacts directory containing:
            - config.yaml: Model configuration
            - model_state_dict: Trained RNN model state dictionary
            - decoder: Character decoder dictionary
            - encoder: Character encoder dictionary
            - demo/: Demo folder with UI components (optional)

    Returns:
        Model: Initialized RNN model instance ready for text generation
    """
    from src.mlflow.model import Model

    logger.info(f"Loading RNN Model from artifacts at: {data_path}")

    from src.utils import load_config

    config_path = os.path.join(data_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    config = load_config(config_path)
    logger.info("Configuration loaded successfully")

    # Set up required RNN model artifacts paths
    model_state_dict_path = os.path.join(data_path, "model_state_dict")
    decoder_path = os.path.join(data_path, "decoder")
    encoder_path = os.path.join(data_path, "encoder")

    # Validate that required artifacts exist
    if not os.path.exists(model_state_dict_path):
        raise FileNotFoundError(
            f"Model state dictionary not found at: {model_state_dict_path}"
        )
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Decoder dictionary not found at: {decoder_path}")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder dictionary not found at: {encoder_path}")

    # Load the all_characters set from the training data
    # For RNN text generation, we need the character set that was used during training
    # This is typically derived from the training text (Shakespeare in this case)
    data_file_path = os.path.join(data_path, "shakespeare.txt")
    if os.path.exists(data_file_path):
        with open(data_file_path, "r", encoding="utf8") as f:
            text = f.read()
        all_chars = sorted(list(set(text)))
        logger.info(
            f"Loaded character set from training data: {len(all_chars)} unique characters"
        )
    else:
        # Fallback: load from encoder keys if data file not available

        encoder_dict = torch.load(encoder_path)
        all_chars = sorted(list(encoder_dict.keys()))
        logger.info(
            f"Character set loaded from encoder: {len(all_chars)} unique characters"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"    

    # Initialize Model
    try:
        model = Model(
            config=config,
            model_state_dict_path=model_state_dict_path,
            decoder_path=decoder_path,
            encoder_path=encoder_path,
            all_chars=all_chars,
            device=device,
        )
        logger.info("RNN Model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize RNN Model: {str(e)}")
        raise RuntimeError(f"RNN Model loading failed: {str(e)}") from e
