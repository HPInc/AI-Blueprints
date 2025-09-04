"""
Logger Service implementation for MLflow model logging.
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

        # Add src directory to path to ensure onnx_utils is found
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from onnx_utils import ModelExportConfig, log_model

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
        import torch
        
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

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading BERT model from: {bert_model_datafabric_path}")
        
            
            # Load the NeMo BERT model into memory
            bert_model =  BERTLMModel.restore_from(bert_model_datafabric_path, strict=False).to(device)
            bert_model.eval() 

            wrapped_model = BERTModelWithHiddenStates(bert_model) #it doesn't have oficial nemo export function so its necessary to recreate the model as torch to use torch conversion
        
            batch_size = 1
            seq_len = 128
            vocab_size = 30522

            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
            token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        
            model_configs = [
                ModelExportConfig(
                    model=wrapped_model,                           # 🚀 Pre-loaded model object!
                    model_name="bert_tourism_onnx",             # ONNX file naming
                    input_sample=(                             
                        input_ids.to(device),
                        attention_mask.to(device),
                        token_type_ids.to(device)
                    ),
                    input_names=["input_ids", "attention_mask", "token_type_ids"],
                    output_names=["embedding"],
                    dynamic_axes={
                        "input_ids": {0: "batch", 1: "sequence"},
                        "attention_mask": {0: "batch", 1: "sequence"},
                        "token_type_ids": {0: "batch", 1: "sequence"},
                        "embedding": {0: "batch_size"}
                    },
                )    
            ]
            
            log_model(
                name=artifact_path,                          
                loader_module="src.mlflow.loader",  
                data_path=temp_dir,                                   
                code_paths=["../src"],                    
                signature=signature,
                models_to_convert_onnx=model_configs,  
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