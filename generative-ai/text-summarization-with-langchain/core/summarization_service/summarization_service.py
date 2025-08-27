"""
Summarization Service implementation for MLflow model logging.

MLflow Registration Layer
- Provides log_model functionality for packaging text summarization models
- Handles artifact organization and temporary directory management
- Uses MLflow's models-from-code approach for deployment
- Manages configuration, secrets, and model assets
"""

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
