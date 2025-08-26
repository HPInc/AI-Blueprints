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
        
        # Prepare conda environment
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
        
        # Log model to MLflow using models-from-code
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            loader_module="core.code_generation_service.code_generation_loader",
            code_paths=["../core", "../src"],
            signature=signature,
            pip_requirements=conda_env,
            

        )
        logger.info("Model and artifacts successfully registered in MLflow using models-from-code.")

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
