"""
Code Generation Service for model registration only.
Business logic has been moved to CodeGenerationModel.

This service handles MLflow registration using models-from-code approach.
"""

import os
import sys
import logging
import tempfile
import yaml
from typing import Dict, Any, Optional

from .code_generation_model import CodeGenerationModel
from .code_generation_loader import save_model

# Set up logger
logger = logging.getLogger(__name__)

class CodeGenerationService:
    """
    Simplified service for registration only - business logic moved to Model.
    """

    def __init__(self, model_instance=None):
        """
        Optional model instance for registration purposes.
        
        Args:
            model_instance: Pre-configured CodeGenerationModel instance (optional)
        """
        self.model = model_instance

    @classmethod
    def log_model(
        cls,
        config_path,
        artifact_path="code_generation_service",
        secrets_dict=None, 
        model_path=None, 
        embedding_model_path=None, 
        delay_async_init=True, 
        demo_folder=None
    ):
        """
        MLflow registration using models-from-code approach.
        
        Args:
            config_path: Path to the configuration file
            artifact_path: Path to store the model artifacts
            secrets_dict: Dict with secrets to persist as YAML (optional)
            model_path: Path to the LLM model file (optional)
            embedding_model_path: Path to the locally saved embedding model directory (optional)
            delay_async_init: If True, delay thread-based component initialization during serialization
            demo_folder: Path to the demo folder (optional)
            
        Returns:
            None
        """
        import mlflow
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec
        
        logger.info("Starting model registration using models-from-code approach")
        
        # Define model input/output schema with all parameters
        input_schema = Schema([
            ColSpec("string", "question"),
            ColSpec("string", "repository_url", required=False),  # Optional repository URL
        ])
        output_schema = Schema([
            ColSpec("string", "result")
        ])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        # Prepare artifacts
        artifacts = {
            "config": config_path
        }

        if secrets_dict:
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
            yaml.safe_dump(secrets_dict, tmp)
            tmp.close()
            artifacts["secrets"] = tmp.name
            logger.info(f"Secrets artifact written to temporary file {tmp.name}")
        
        if model_path:
            artifacts["models"] = model_path
            
        # Add demo folder to artifacts if provided
        if demo_folder:
            artifacts["demo"] = demo_folder
            
        # Add embedding model path to artifacts if provided and exists
        if embedding_model_path and os.path.exists(embedding_model_path):
            artifacts["embedding_model"] = embedding_model_path
            logger.info(f"Using local embedding model from: {embedding_model_path}")
        else:
            logger.warning("No local embedding model path provided or path doesn't exist. " 
                         "The service will download the embedding model during initialization.")
        
        # Create temp directory for artifacts (similar to working PR #227)
        import tempfile
        import shutil
        
        temp_base = tempfile.gettempdir()
        temp_dir = os.path.join(temp_base, "model_artifacts")
        
        # Clean slate for deterministic results
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        try:
            logger.info(f"Organizing artifacts in temp directory: {temp_dir}")
            
            # Copy config to temp directory -> /artifacts/data/config.yaml
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at: {config_path}")
            shutil.copy2(config_path, os.path.join(temp_dir, "config.yaml"))
            logger.info(f"Copied config from {config_path} to temp directory")
            
            # Copy demo folder if provided -> /artifacts/data/demo/
            if demo_folder and os.path.exists(demo_folder):
                shutil.copytree(demo_folder, os.path.join(temp_dir, "demo"))
                logger.info(f"Copied demo folder from {demo_folder}")
            else:
                logger.info("Demo folder not provided or doesn't exist - skipping")
            
            # Handle model files if provided -> /artifacts/data/models/
            if model_path and os.path.exists(model_path):
                models_dir = os.path.join(temp_dir, "models")
                os.makedirs(models_dir, exist_ok=True)
                if os.path.isfile(model_path):
                    shutil.copy2(model_path, models_dir)
                    logger.info(f"Copied model file: {os.path.basename(model_path)}")
                else:
                    # For model directories, copy contents
                    for item in os.listdir(model_path):
                        item_path = os.path.join(model_path, item)
                        if os.path.isfile(item_path):
                            shutil.copy2(item_path, models_dir)
                        else:
                            shutil.copytree(item_path, os.path.join(models_dir, item))
                    logger.info(f"Copied model directory contents: {model_path}")
            else:
                logger.info("Model path not provided or doesn't exist - skipping")
            
            # Handle embedding model if provided -> /artifacts/data/embedding_model/
            if embedding_model_path and os.path.exists(embedding_model_path):
                shutil.copytree(embedding_model_path, os.path.join(temp_dir, "embedding_model"))
                logger.info(f"Copied embedding model from {embedding_model_path}")
            else:
                logger.info("Embedding model path not provided or doesn't exist - will download during runtime")
            
            # Log model to MLflow using models-from-code with data_path
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                loader_module="core.code_generation_service.code_generation_loader",
                data_path=temp_dir,  # Use data_path instead of artifacts
                code_paths=["../core", "../src"],
                signature=signature,
                pip_requirements="../requirements.txt"
            )
            logger.info("Model and artifacts successfully registered in MLflow using models-from-code.")
            
        except Exception as e:
            logger.error(f"Error during model logging: {str(e)}")
            raise
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary directory")

    def get_model_info(self):
        """
        Get information about the model, if available.
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {"error": "No model instance available"}
            
        try:
            # Get model info from the model instance
            if hasattr(self.model, 'llm') and self.model.llm:
                context_window = getattr(self.model.llm, '_context_window', 'unknown')
                model_type = type(self.model.llm).__name__
                
                # Get additional info based on model type
                additional_info = {}
                if hasattr(self.model.llm, 'model_path'):
                    additional_info['model_path'] = self.model.llm.model_path
                if hasattr(self.model.llm, 'repo_id'):
                    additional_info['repo_id'] = self.model.llm.repo_id
                    
                return {
                    "model_type": model_type,
                    "context_window": context_window,
                    "additional_info": additional_info,
                    "success": True
                }
            else:
                return {"error": "No LLM loaded in model instance"}
        except Exception as e:
            return {"error": f"Error retrieving model info: {str(e)}"}

    def predict(self, model_input: Dict[str, Any], params=None):
        """
        Wrapper for model prediction - delegates to model instance.
        
        Args:
            model_input: Input data for code generation
            params: Additional parameters
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("No model instance available for prediction")
            
        return self.model.predict(model_input, params)
