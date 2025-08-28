"""
Logger Service implementation for MLflow Iris Classification model logging.

MLflow Registration Layer
- Provides log_model functionality for iris classification models
- Handles artifact organization and temporary directory management 
- Uses MLflow's models-from-code approach for deployment
- Manages configuration and demo assets for classification use case
"""
import os
import uuid
import base64
import logging
import shutil
from typing import Dict, Any, List
import yaml
import tempfile
import pandas as pd

# Set up logger
logger = logging.getLogger(__name__)

class Logger:
    """
    Logger Service for MLflow iris classification model logging.
    This class provides the log_model functionality for packaging iris
    classification models with their configuration and demo assets.
    """
    
    def __init__(self):
        """Initialize the logger service for logging purposes."""
        logger.info("Logger initialized for MLflow model logging")

    @classmethod
    def log_model(
        cls,
        signature,
        artifact_path="AIStudio-Model",
        config_path="configs/config.yaml",
        demo_folder=None
    ):
        """
        Log iris classification model using models-from-code approach with simplified directory structure.
        
        This implementation uses MLflow's models-from-code approach for iris classification models.
        The classification model only needs configuration and optional demo assets.
        
        Final MLflow structure achieved:
        /artifacts/
          └── data/                    # MLflow automatically created
              ├── config.yaml          # Configuration with dataset URL
              └── demo/                # UI components (optional)
        
        Args:
            signature: MLflow ModelSignature defining input/output schema for the iris classification model
            artifact_path: Path to store the model artifacts
            config_path: Path to the configuration file
            demo_folder: Path to the demo folder (optional)
            
        Returns:
            None
        """
        import mlflow
        import tempfile
        import shutil
        import os
        import yaml
        
        # Create temp directory
        temp_base = tempfile.gettempdir()
        temp_dir = os.path.join(temp_base, "model_artifacts")
        
        # Clean slate for deterministic results
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        try:
            logger.info(f"Organizing artifacts in temp directory: {temp_dir}")
            # Organize temp directory for clean final structure
            # MLflow will place this under /artifacts/data/ automatically
            
            # ✅ Config at root -> /artifacts/data/config.yaml
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at: {config_path}")
            shutil.copy2(config_path, os.path.join(temp_dir, "config.yaml"))
            logger.info(f"Copied config from {config_path} to temp directory")
            
            # ✅ Demo folder -> /artifacts/data/demo/ (optional)
            if demo_folder and os.path.exists(demo_folder):
                shutil.copytree(demo_folder, os.path.join(temp_dir, "demo"))
                logger.info(f"Copied demo folder from {demo_folder}")
            else:
                logger.info("Demo folder not provided or doesn't exist - skipping")
            
            # Log the iris classification model using models-from-code approach
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,                          
                loader_module="src.mlflow.loader",  
                data_path=temp_dir,                                   
                code_paths=["../src"],                    
                signature=signature,
                pip_requirements="../requirements.txt"
            )
        except Exception as e:
            logger.error(f"Error during model logging: {str(e)}")
            raise
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary directory")