"""
Simplified AgenticService for registration only.

Registration Layer
- Handles MLflow model registration using models-from-code approach
- Business logic moved to AgenticModel class
- Provides clean interface for model logging and deployment
"""

import logging
import os
from typing import Dict, Any, Optional

# Set up logger
logger = logging.getLogger(__name__)


class AgenticService:
    """
    Simplified service for model registration only - business logic moved to AgenticModel.
    
    This service class handles MLflow registration using the models-from-code approach,
    eliminating the problematic serialization issues of the legacy PythonModel pattern.
    """

    def __init__(self, model_instance=None):
        """
        Optional model instance for registration purposes.
        
        Args:
            model_instance: Optional AgenticModel instance (primarily for testing)
        """
        self.model = model_instance

    @classmethod
    def log_model(
        cls,
        model_name: str,
        model_artifacts: Dict[str, str],
        **kwargs
    ):
        """
        Log model using models-from-code approach with proper artifact organization.
        
        Args:
            model_name: Name for the registered model (used as artifact_path)
            model_artifacts: Dictionary containing model and memory paths
                Expected keys:
                - 'model_path': Path to LLM model file
                - 'memory_path': Path to memory storage directory
            **kwargs: Additional arguments (preserved for compatibility)
        """
        import mlflow
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec
        import tempfile
        import shutil
        import os
        
        # Extract paths from model_artifacts
        model_path = model_artifacts.get("model_path")
        memory_path = model_artifacts.get("memory_path")
        
        # Set up defaults
        config_path = "configs/config.yaml"
        docs_path = "data/input"
        demo_folder = None  # Not used in current implementation
        
        # Define model input/output schema
        input_schema = Schema([
            ColSpec("string", "topic"),
            ColSpec("string", "question"), 
            ColSpec("string", "input_text")
        ])
        output_schema = Schema([
            ColSpec("string", "answer"),
            ColSpec("string", "messages")
        ])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        # Create temp directory
        temp_base = tempfile.gettempdir()
        temp_dir = os.path.join(temp_base, "agentic_model_artifacts")
        
        # Clean slate for deterministic results
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        try:
            logger.info(f"Organizing artifacts in temp directory: {temp_dir}")
            
            # ✅ Config at root -> /artifacts/data/config.yaml
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at: {config_path}")
            shutil.copy2(config_path, os.path.join(temp_dir, "config.yaml"))
            logger.info(f"Copied config from {config_path} to temp directory")
            
            # ✅ Create data subdirectory -> /artifacts/data/data/
            data_temp_dir = os.path.join(temp_dir, "data")
            os.makedirs(data_temp_dir, exist_ok=True)
            
            # Copy documents to data subdirectory
            if docs_path and os.path.exists(docs_path):
                for item in os.listdir(docs_path):
                    item_path = os.path.join(docs_path, item)
                    if os.path.isfile(item_path):
                        shutil.copy2(item_path, data_temp_dir)
                        logger.info(f"Copied document: {item}")
                    elif os.path.isdir(item_path):
                        shutil.copytree(item_path, os.path.join(data_temp_dir, item))
                        logger.info(f"Copied document directory: {item}")
            else:
                logger.info("Documents path not provided or doesn't exist - skipping")
            
            # ✅ Handle memory directory -> /artifacts/data/memory/
            if memory_path and os.path.exists(memory_path):
                shutil.copytree(memory_path, os.path.join(temp_dir, "memory"))
                logger.info(f"Copied memory directory from {memory_path}")
            else:
                # Create empty memory directory
                os.makedirs(os.path.join(temp_dir, "memory"), exist_ok=True)
                logger.info("Created empty memory directory")
                
            # ✅ Handle model files -> /artifacts/data/models/
            if model_path and os.path.exists(model_path):
                models_temp_dir = os.path.join(temp_dir, "models")
                os.makedirs(models_temp_dir, exist_ok=True)
                if os.path.isfile(model_path):
                    shutil.copy2(model_path, os.path.join(models_temp_dir, os.path.basename(model_path)))
                    logger.info(f"Copied model file: {os.path.basename(model_path)}")
                else:
                    shutil.copytree(model_path, models_temp_dir, dirs_exist_ok=True)
                    logger.info(f"Copied model directory: {model_path}")
            else:
                logger.info("Model path not provided or doesn't exist - skipping")
            
            # ✅ Demo folder -> /artifacts/data/demo/
            if demo_folder and os.path.exists(demo_folder):
                shutil.copytree(demo_folder, os.path.join(temp_dir, "demo"))
                logger.info(f"Copied demo folder from {demo_folder}")
            else:
                logger.info("Demo folder not provided or doesn't exist - skipping")
            
            mlflow.pyfunc.log_model(
                artifact_path=model_name,  # Use model_name as artifact_path
                loader_module="core.agentic_service.agentic_loader",
                data_path=temp_dir,
                code_paths=["../core", "../src"],
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

    @classmethod
    def load_model(cls, model_uri: str):
        """
        Load a registered model for inference.
        
        Args:
            model_uri: MLflow model URI
            
        Returns:
            Loaded model instance
        """
        import mlflow
        
        logger.info(f"Loading model from URI: {model_uri}")
        
        try:
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Model loaded successfully")
            return loaded_model
        except Exception as e:
            logger.error(f"Failed to load model from {model_uri}: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}") from e
