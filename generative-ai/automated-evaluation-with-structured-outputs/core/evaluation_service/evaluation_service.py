"""
Evaluation Service implementation for MLflow model logging.

MLflow Registration Layer
- Provides log_model functionality for packaging evaluation models
- Handles artifact organization and model registration
- Uses MLflow's models-from-code approach for deployment
- Manages configuration and LLaMA model artifacts
"""

import os
import logging
import tempfile
import shutil
from typing import Dict, Any, Optional
import pandas as pd

# Set up logger
logger = logging.getLogger(__name__)


class EvaluationService:
    """
    Evaluation Service for MLflow model logging.
    This class provides the log_model functionality for packaging 
    automated evaluation models with structured outputs.
    """
    
    def __init__(self, model_instance=None):
        """Initialize the evaluation service for logging purposes."""
        self.model = model_instance
        logger.info("EvaluationService initialized for MLflow model logging")

    @classmethod
    def log_model(
        cls,
        model_name: str,
        llama_model_path: str,
        config_path: str,
        experiment_name: str = "EvaluationExperiment"
    ):
        """
        Log model using models-from-code approach.
        
        This implementation uses MLflow's models-from-code approach with proper
        artifact organization for automated evaluation models.
        
        Args:
            model_name: Name for the registered model
            llama_model_path: Path to the LLaMA model file
            config_path: Path to the configuration file
            experiment_name: MLflow experiment name
            
        Returns:
            None
        """
        import mlflow
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec, ParamSchema, ParamSpec, DataType
        
        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)
        
        # Define model input/output schema based on original EvaluatorModel
        input_schema = Schema([
            ColSpec("string", "title"),
            ColSpec("string", "abstract")
        ])
        
        output_schema = Schema([
            ColSpec("string", "title"),
            ColSpec("string", "abstract"),
            ColSpec("integer", "Originality"),
            ColSpec("integer", "Clarity"),
            ColSpec("integer", "Relevance"),
            ColSpec("integer", "Feasibility"),
            ColSpec("integer", "Impact"),
            ColSpec("integer", "TotalScore")
        ])
        
        # Define parameters schema
        params_schema = ParamSchema([
            ParamSpec("key_column",  DataType.string,  "title"),
            ParamSpec("eval_column", DataType.string,  "abstract"),
            ParamSpec("criteria",    DataType.string,  '{"Originality":20,"Clarity":20,"Relevance":20,"Feasibility":20,"Impact":20}'),
        ])
        
        signature = ModelSignature(
            inputs=input_schema, 
            outputs=output_schema, 
            params=params_schema
        )
        
        # Create temporary directory for artifacts
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Organizing artifacts in temporary directory: {temp_dir}")
            
            # Copy configuration file
            if os.path.exists(config_path):
                shutil.copy2(config_path, os.path.join(temp_dir, "config.yaml"))
                logger.info(f"Configuration copied from {config_path}")
            
            # Create artifacts dictionary for MLflow
            artifacts = {
                "llama_model_path": llama_model_path,
                "config_path": config_path,
                "demo": "../demo"
            }
            
            # Define code paths for models-from-code
            code_paths = [
                "core/",
                "src/"
            ]
            
            # Create conda environment specification
            conda_env = {
                "name": "evaluation_env",
                "channels": ["conda-forge", "defaults"],
                "dependencies": [
                    "python=3.11",
                    "pip",
                    {
                        "pip": [
                            "mlflow==3.1.0",
                            "pandas>=2.0.0",
                            "llama-cpp-python>=0.2.0",
                            "pyyaml>=6.0",
                        ]
                    }
                ]
            }
            
            # Log model using models-from-code approach
            mlflow.models.save_model(
                path=temp_dir,
                loader_module="core.evaluation_service.evaluation_loader",
                data_path=temp_dir,
                signature=signature,
                conda_env=conda_env,
                code_paths=code_paths
            )
            
            # Register the model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
            mlflow.register_model(model_uri, model_name)
            
            logger.info(f"Model '{model_name}' logged and registered successfully")
    
    def predict(self, model_input: pd.DataFrame, params: dict = None) -> pd.DataFrame:
        """
        Wrapper predict method for direct service usage.
        Delegates to the underlying model instance.
        """
        if self.model is None:
            raise ValueError("No model instance available for prediction")
        
        return self.model.predict(model_input, params)
