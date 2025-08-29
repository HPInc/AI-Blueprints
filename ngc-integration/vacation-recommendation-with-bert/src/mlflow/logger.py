"""
Logger Service implementation for BERT Tourism Recommendation MLflow model logging.

MLflow Registration Layer
- Provides log_model functionality for BERT tourism models
- Handles artifact organization and temporary directory management
- Uses MLflow's models-from-code approach for deployment
- Manages embeddings, corpus, tokenizer, BERT model, and demo assets
- Supports ONNX model export integration
"""
import os
import uuid
import base64
import logging
import shutil
from typing import Dict, Any, List, Optional
import yaml
import tempfile
import pandas as pd

# Set up logger
logger = logging.getLogger(__name__)

class Logger:
    """
    Logger Service for BERT Tourism Recommendation MLflow model logging.
    This class provides the log_model functionality for packaging BERT-based
    vacation recommendation models with precomputed embeddings and corpus data.
    """

    def __init__(self):
        """Initialize the logger service for logging purposes."""
        logger.info("BERT Tourism Logger initialized for MLflow model logging")

    @classmethod
    def log_model(
        cls,
        signature,
        artifact_path="BERT_Tourism_Model",
        config_path="configs/config.yaml",
        corpus_path=None,
        embeddings_path=None,
        tokenizer_dir=None,
        bert_model_path=None,
        demo_path=None,
        models_to_convert_onnx=None,
        **kwargs
    ):
        """
        Log BERT Tourism model using models-from-code approach with ONNX export support.

        This implementation uses MLflow's models-from-code approach for BERT tourism
        recommendation models, organizing all necessary artifacts including precomputed
        embeddings, corpus data, tokenizer, and BERT model files.

        Final MLflow structure achieved:
        /artifacts/
          └── data/                    # MLflow automatically created
              ├── config.yaml          # Configuration
              ├── corpus.csv           # Tourism corpus data
              ├── embeddings.csv       # Precomputed embeddings
              ├── tokenizer/           # BERT tokenizer directory
              ├── bert_model.nemo      # BERT model (optional - can use external path)
              ├── demo/                # UI components (optional)
              └── onnx_models/         # ONNX exported models (optional)

        Args:
            signature: MLflow ModelSignature defining input/output schema
            artifact_path: Path to store the model artifacts
            config_path: Path to the configuration file
            corpus_path: Path to the tourism corpus CSV file
            embeddings_path: Path to the precomputed embeddings CSV file
            tokenizer_dir: Path to the BERT tokenizer directory
            bert_model_path: Path to the BERT model file (optional - can use external)
            demo_path: Path to the demo folder (optional)
            models_to_convert_onnx: List of ModelExportConfig for ONNX export (optional)
            **kwargs: Additional arguments for backward compatibility

        Returns:
            None
        """
        import mlflow

        # Create temp directory
        temp_base = tempfile.gettempdir()
        temp_dir = os.path.join(temp_base, "bert_tourism_artifacts")

        # Clean slate for deterministic results
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        try:
            logger.info(f"Organizing BERT Tourism artifacts in temp directory: {temp_dir}")

            # ✅ Config at root -> /artifacts/data/config.yaml
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at: {config_path}")
            shutil.copy2(config_path, os.path.join(temp_dir, "config.yaml"))
            logger.info(f"Copied config from {config_path} to temp directory")

            # ✅ Corpus data -> /artifacts/data/corpus.csv
            if corpus_path and os.path.exists(corpus_path):
                shutil.copy2(corpus_path, os.path.join(temp_dir, "corpus.csv"))
                logger.info(f"Copied corpus data from {corpus_path}")
            else:
                logger.warning("Corpus path not provided or doesn't exist")

            # ✅ Embeddings data -> /artifacts/data/embeddings.csv
            if embeddings_path and os.path.exists(embeddings_path):
                shutil.copy2(embeddings_path, os.path.join(temp_dir, "embeddings.csv"))
                logger.info(f"Copied embeddings from {embeddings_path}")
            else:
                logger.warning("Embeddings path not provided or doesn't exist")

            # ✅ Tokenizer -> /artifacts/data/tokenizer/
            if tokenizer_dir and os.path.exists(tokenizer_dir):
                shutil.copytree(tokenizer_dir, os.path.join(temp_dir, "tokenizer"))
                logger.info(f"Copied tokenizer from {tokenizer_dir}")
            else:
                logger.warning("Tokenizer directory not provided or doesn't exist")

            # ✅ BERT Model (optional - can use external path) -> /artifacts/data/bert_model.nemo
            if bert_model_path and os.path.exists(bert_model_path):
                if os.path.isfile(bert_model_path):
                    # Copy single model file
                    model_filename = os.path.basename(bert_model_path)
                    shutil.copy2(bert_model_path, os.path.join(temp_dir, model_filename))
                    logger.info(f"Copied BERT model file: {model_filename}")
                else:
                    # Copy model directory
                    shutil.copytree(bert_model_path, os.path.join(temp_dir, "bert_model"))
                    logger.info(f"Copied BERT model directory from {bert_model_path}")
            else:
                logger.info("BERT model path not provided - will use external path reference")

            # ✅ Demo folder -> /artifacts/data/demo/
            if demo_path and os.path.exists(demo_path):
                shutil.copytree(demo_path, os.path.join(temp_dir, "demo"))
                logger.info(f"Copied demo folder from {demo_path}")
            else:
                logger.info("Demo folder not provided or doesn't exist - skipping")

            # ✅ Handle ONNX model export if specified
            if models_to_convert_onnx:
                logger.info("Processing ONNX model export configurations")
                onnx_temp_dir = os.path.join(temp_dir, "onnx_models")
                os.makedirs(onnx_temp_dir, exist_ok=True)

                # Import ONNX export functionality
                try:
                    from ..onnx_utils import log_model as log_model_with_onnx
                    
                    # Use the custom ONNX logging function if available
                    log_model_with_onnx(
                        artifact_path=artifact_path,
                        python_model=None,  # Will be handled by models-from-code approach
                        artifacts={
                            "config": os.path.join(temp_dir, "config.yaml"),
                            "corpus_path": os.path.join(temp_dir, "corpus.csv"),
                            "embeddings_path": os.path.join(temp_dir, "embeddings.csv"),
                            "tokenizer_dir": os.path.join(temp_dir, "tokenizer"),
                        },
                        signature=signature,
                        models_to_convert_onnx=models_to_convert_onnx,
                        pip_requirements="../requirements.txt",
                        code_paths=["../src"],
                        loader_module="src.mlflow.loader",
                        data_path=temp_dir
                    )
                    logger.info("ONNX model export completed")
                    return  # Early return since ONNX export handles the logging
                
                except ImportError as e:
                    logger.warning(f"ONNX export functionality not available: {e}")
                    logger.info("Proceeding with standard MLflow logging")

            # Standard MLflow logging using models-from-code approach
            mlflow.pyfunc.log_model(
                name=artifact_path,
                loader_module="src.mlflow.loader",
                data_path=temp_dir,
                code_paths=["../src"],
                signature=signature,
                pip_requirements="../requirements.txt"
            )
            logger.info(f"Successfully logged BERT Tourism model with artifact path: {artifact_path}")
            
        except Exception as e:
            logger.error(f"Error during BERT Tourism model logging: {str(e)}")
            raise
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary directory")