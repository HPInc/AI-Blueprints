"""
Logger Service implementation for MLflow model logging.

MLflow Registration Layer
- Provides log_model functionality for models
- Handles artifact organization and temporary directory management
- Uses MLflow's models-from-code approach for deployment
- Manages configuration, documents, secrets, and demo assets
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
    Logger Service for MLflow model logging.
    This class provides the log_model functionality for packaging RAG-based
    conversational AI with document retrieval capabilities.
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
        docs_path="data/",
        secrets_dict=None,
        model_path=None,
        demo_folder=None
    ):
        """
        Log model using refined models-from-code approach with elegant directory structure.
        
        This implementation uses MLflow's models-from-code approach exclusively with proper
        temp directory naming to avoid redundant nesting while maintaining full MLflow 3.1.0 compatibility.
        
        Final MLflow structure achieved:
        /artifacts/
          └── data/                    # MLflow automatically created
              ├── config.yaml          # Configuration
              ├── data/                # Documents directory (PDFs, etc.)
              ├── demo/                # UI components  
              ├── models/              # Model files (optional)
              └── secrets.yaml         # Secrets (optional)
        
        Args:
            signature: MLflow ModelSignature defining input/output schema for the model
            artifact_path: Path to store the model artifacts
            config_path: Path to the configuration file
            docs_path: Path to the documents directory
            secrets_dict: Dict with secrets to persist as YAML (optional)
            model_path: Path to the model file (optional)
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
            logger.info("data folder not provided or doesn't exist - skipping")
            
            
            # ✅ Demo folder -> /artifacts/data/demo/
            if demo_folder and os.path.exists(demo_folder):
                shutil.copytree(demo_folder, os.path.join(temp_dir, "demo"))
                logger.info(f"Copied demo folder from {demo_folder}")
            else:
                logger.info("Demo folder not provided or doesn't exist - skipping")
            
            # ✅ Handle secrets -> /artifacts/data/secrets.yaml
            if secrets_dict:
                with open(os.path.join(temp_dir, "secrets.yaml"), 'w') as f:
                    yaml.safe_dump(secrets_dict, f)
                logger.info("Created secrets.yaml in temp directory")
                    
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
            
            mlflow.pyfunc.log_model(
                name=artifact_path,                          
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

import os
import logging
import shutil
from typing import Dict, Any, Optional
import yaml
import tempfile
import pandas as pd

# Set up logger
logger = logging.getLogger(__name__)


class SummarizationService:
    """
    Summarization Service for MLflow model logging.
    This class provides the log_model functionality for packaging text summarization
    models using different LLM backends (local, HuggingFace local, HuggingFace cloud).
    """
    
    def __init__(self, model_instance=None):
        """
        Initialize the service.
        
        Args:
            model_instance: Optional SummarizationModel instance for direct usage
        """
        self.model = model_instance
    
    @classmethod
    def log_model(
        cls, 
        artifact_path: str, 
        config_path: str,
        secrets_dict: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None, 
        demo_folder: Optional[str] = None,
        **kwargs
    ):
        """
        Log the model to MLflow using models-from-code approach.
        
        Args:
            artifact_path: Path to store the model artifacts in MLflow
            config_path: Path to the configuration file
            secrets_dict: Dict with secrets to persist as YAML (optional)
            model_path: Path to the model file (optional, for local models)
            demo_folder: Path to the demo folder (optional)
            **kwargs: Additional arguments passed to mlflow.models.save_model
            
        Returns:
            None
        """
        import mlflow
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec
        
        try:
            logger.info(f"Starting model logging for summarization service")
            logger.info(f"Artifact path: {artifact_path}")
            logger.info(f"Config path: {config_path}")
            
            # Create demo folder if specified and doesn't exist
            if demo_folder and not os.path.exists(demo_folder):
                os.makedirs(demo_folder, exist_ok=True)
                logger.info(f"Created demo folder: {demo_folder}")
                
            # Define model input/output schema
            input_schema = Schema([
                ColSpec("string", "text")
            ])
            output_schema = Schema([
                ColSpec("string", "summary")
            ])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)
            logger.info("Model signature created")
            
            # Create temporary directory for organizing artifacts
            with tempfile.TemporaryDirectory() as temp_dir:
                data_path = os.path.join(temp_dir, "model_data")
                os.makedirs(data_path, exist_ok=True)
                
                # Copy configuration file
                config_dest = os.path.join(data_path, "config.yaml")
                shutil.copy2(config_path, config_dest)
                logger.info(f"Configuration copied to: {config_dest}")
                
                # Handle secrets if provided
                if secrets_dict:
                    secrets_dest = os.path.join(data_path, "secrets.yaml")
                    with open(secrets_dest, 'w') as f:
                        yaml.safe_dump(secrets_dict, f)
                    logger.info(f"Secrets written to: {secrets_dest}")
                
                # Handle model files if provided (for local models)
                if model_path and os.path.exists(model_path):
                    model_dest_dir = os.path.join(data_path, "model")
                    if os.path.isfile(model_path):
                        # Single model file
                        os.makedirs(model_dest_dir, exist_ok=True)
                        model_dest = os.path.join(model_dest_dir, os.path.basename(model_path))
                        shutil.copy2(model_path, model_dest)
                        logger.info(f"Model file copied to: {model_dest}")
                    elif os.path.isdir(model_path):
                        # Model directory
                        shutil.copytree(model_path, model_dest_dir)
                        logger.info(f"Model directory copied to: {model_dest_dir}")
                
                # Handle demo folder if provided
                if demo_folder and os.path.exists(demo_folder):
                    demo_dest = os.path.join(data_path, "demo")
                    shutil.copytree(demo_folder, demo_dest)
                    logger.info(f"Demo folder copied to: {demo_dest}")
                
                # Prepare conda environment
                conda_env = {
                    'channels': ['defaults', 'conda-forge', 'huggingface'],
                    'dependencies': [
                        'python=3.10',
                        'pip',
                        {
                            'pip': [
                                'mlflow>=2.8.0',
                                'langchain>=0.1.0',
                                'langchain-community>=0.0.10',
                                'langchain-core>=0.1.0',
                                'langchain-huggingface>=0.0.3',
                                'transformers>=4.30.0',
                                'torch>=2.0.0',
                                'pandas>=1.5.0',
                                'PyYAML>=6.0',
                                'llama-cpp-python>=0.2.0',
                                'sentence-transformers>=2.2.0',
                                'huggingface_hub>=0.16.0'
                            ]
                        }
                    ],
                    'name': 'summarization_env'
                }
                
                # Code paths to include
                code_paths = ["../core", "../src"]
                
                # Additional kwargs handling
                extra_kwargs = {}
                if 'pip_requirements' in kwargs:
                    extra_kwargs['pip_requirements'] = kwargs['pip_requirements']
                elif not kwargs.get('disable_default_requirements', False):
                    # Use requirements.txt if available
                    requirements_path = os.path.join(os.path.dirname(config_path), "..", "requirements.txt")
                    if os.path.exists(requirements_path):
                        extra_kwargs['pip_requirements'] = requirements_path
                
                # Log model using models-from-code approach
                mlflow.pyfunc.log_model(
                    name=artifact_path,
                    loader_module="core.summarization_service.summarization_loader",
                    data_path=data_path,
                    code_paths=code_paths,
                    signature=signature,
                    conda_env=conda_env
                    
                )
                
                logger.info("Model successfully logged to MLflow using models-from-code approach")
                
        except Exception as e:
            logger.error(f"Error logging model: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def predict(self, model_input: Dict[str, Any]) -> pd.DataFrame:
        """
        Make predictions using the loaded model instance.
        This method is for direct usage when the service has a model_instance.
        
        Args:
            model_input: Input data for summarization
            
        Returns:
            DataFrame with predictions
        """
        if not self.model:
            raise RuntimeError("No model instance available. Use log_model for MLflow registration.")
        
        return self.model.predict(model_input)

    @classmethod
    def create_sample_input_output(cls) -> tuple:
        """
        Create sample input and output for signature inference.
        
        Returns:
            Tuple of (sample_input, sample_output) for MLflow signature
        """
        sample_input = pd.DataFrame([{"text": "This is a sample text for summarization."}])
        sample_output = pd.DataFrame([{"summary": "This is a sample summary."}])
        return sample_input, sample_output
