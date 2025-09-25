"""
Logger Service implementation for MLflow model logging for Movie Recommendation System.

MLflow Registration Layer
- Provides log_model functionality for movie recommendation models
- Handles artifact organization and temporary directory management
- Uses MLflow's models-from-code approach for deployment
- Manages configuration, training data, movie titles, and demo assets
"""

import os
import logging
import shutil
import tempfile
from typing import Dict, Any, Optional

# Set up logger
logger = logging.getLogger(__name__)


class Logger:
    """
    Logger Service for MLflow model logging.
    This class provides the log_model functionality for packaging movie recommendation
    models with training data and movie titles.
    """

    def __init__(self):
        """Initialize the logger service for logging purposes."""
        logger.info("Logger initialized for MLflow model logging")

    @classmethod
    def log_model(
        cls,
        train_data_matrix_path: str,
        movie_titles_path: str,
        config_path: str = "configs/config.yaml",
        demo_folder: Optional[str] = None,
        artifact_path: str = "AIStudio-Model",
        signature=None,
    ):
        """
        Log model using models-from-code approach for movie recommendation system.

        This implementation uses MLflow's models-from-code approach with proper
        temp directory management for the recommendation system artifacts.

        Final MLflow structure achieved:
        /artifacts/
          └── data/                    # MLflow automatically created
              ├── config.yaml          # Configuration
              ├── train_data_matrix.npy # Training data matrix
              ├── movie_titles.csv     # Movie titles and metadata
              └── demo/                # UI components (optional)

        Args:
            train_data_matrix_path: Path to training data matrix (.npy file)
            movie_titles_path: Path to movie titles CSV file
            config_path: Path to the configuration file
            demo_folder: Path to the demo folder (optional)
            artifact_path: Path to store the model artifacts
            signature: MLflow ModelSignature defining input/output schema for the model

        Returns:
            None
        """
        import mlflow

        # Create temp directory
        temp_base = tempfile.gettempdir()
        temp_dir = os.path.join(temp_base, "movie_model_artifacts")

        # Clean slate for deterministic results
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        try:
            logger.info(
                f"Organizing movie recommendation artifacts in temp directory: {temp_dir}"
            )

            # ✅ Config at root -> /artifacts/data/config.yaml
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at: {config_path}")
            shutil.copy2(config_path, os.path.join(temp_dir, "config.yaml"))
            logger.info(f"Copied config from {config_path} to temp directory")

            # ✅ Training data matrix -> /artifacts/data/train_data_matrix.npy
            if not os.path.exists(train_data_matrix_path):
                raise FileNotFoundError(
                    f"Training data matrix not found at: {train_data_matrix_path}"
                )
            shutil.copy2(
                train_data_matrix_path, os.path.join(temp_dir, "train_data_matrix.npy")
            )
            logger.info(f"Copied training data matrix from {train_data_matrix_path}")

            # ✅ Movie titles -> /artifacts/data/movie_titles.csv
            if not os.path.exists(movie_titles_path):
                raise FileNotFoundError(
                    f"Movie titles file not found at: {movie_titles_path}"
                )
            shutil.copy2(movie_titles_path, os.path.join(temp_dir, "movie_titles.csv"))
            logger.info(f"Copied movie titles from {movie_titles_path}")

            # ✅ Demo folder -> /artifacts/data/demo/
            if demo_folder and os.path.exists(demo_folder):
                shutil.copytree(demo_folder, os.path.join(temp_dir, "demo"))
                logger.info(f"Copied demo folder from {demo_folder}")
            else:
                logger.info("Demo folder not provided or doesn't exist - skipping")

            mlflow.pyfunc.log_model(
                name=artifact_path,
                loader_module="src.mlflow.loader",
                data_path=temp_dir,
                code_paths=["../src"],
                signature=signature,
                pip_requirements="../requirements.txt",
            )

            logger.info("✅ Movie recommendation model logged successfully to MLflow")
        except Exception as e:
            logger.error(f"Error during model logging: {str(e)}")
            raise
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary directory")
