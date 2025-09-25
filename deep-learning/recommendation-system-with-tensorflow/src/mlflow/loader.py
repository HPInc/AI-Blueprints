"""
MLflow models-from-code loader module for Movie Recommendation System.
This module provides the _load_pyfunc function required by MLflow's models-from-code approach.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

# Set up logger
logger = logging.getLogger(__name__)


def _load_pyfunc(data_path: str):
    """
    MLflow models-from-code loader function.
    Called by MLflow to load the movie recommendation model from artifacts.

    Args:
        data_path: Path to model artifacts directory containing:
            - config.yaml: Model configuration
            - train_data_matrix.npy: Training data matrix
            - movie_titles.csv: Movie titles and metadata
            - demo/: Demo folder with UI components (optional)

    Returns:
        Model: Initialized movie recommendation model instance ready for prediction
    """
    from src.mlflow.model import Model

    logger.info(f"Loading Movie Recommendation Model from artifacts at: {data_path}")

    from src.utils import load_config

    # Load configuration
    config_path = os.path.join(data_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    config = load_config(config_path)
    logger.info("Configuration loaded successfully")

    # Load training data matrix
    train_data_matrix_path = os.path.join(data_path, "train_data_matrix.npy")
    if not os.path.exists(train_data_matrix_path):
        raise FileNotFoundError(
            f"Training data matrix not found at: {train_data_matrix_path}"
        )

    train_data_matrix = np.load(train_data_matrix_path)
    logger.info(f"Training data matrix loaded: shape {train_data_matrix.shape}")

    # Load movie titles
    movie_titles_path = os.path.join(data_path, "movie_titles.csv")
    if not os.path.exists(movie_titles_path):
        raise FileNotFoundError(f"Movie titles file not found at: {movie_titles_path}")

    movie_titles = pd.read_csv(movie_titles_path)
    logger.info(f"Movie titles loaded: {len(movie_titles)} movies")

    # Initialize Model
    try:
        model = Model(
            train_data_matrix=train_data_matrix,
            movie_titles=movie_titles,
            config=config,
        )
        logger.info("Movie Recommendation Model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e
