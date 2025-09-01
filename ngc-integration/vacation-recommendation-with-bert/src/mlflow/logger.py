""""""

Logger Service implementation for MLflow model logging.Logger Service implementation for MLflow model logging.



MLflow Registration LayerMLflow Registration Layer

- Provides log_model functionality for models- Provides log_model functionality for models

- Handles artifact organization and temporary directory management- Handles artifact organization and temporary directory management

- Uses MLflow's models-from-code approach for deployment- Uses MLflow's models-from-code approach for deployment

- Manages configuration, documents, secrets, and demo assets- Manages configuration, embeddings, corpus, tokenizer, model files, and demo assets

""""""

import osimport os

import uuidimport uuid

import base64import base64

import loggingimport logging

import shutilimport shutil

from typing import Dict, Any, Listfrom typing import Dict, Any, List

import yamlimport yaml

import tempfileimport tempfile

import pandas as pdimport pandas as pd



# Set up logger# Set up logger

logger = logging.getLogger(__name__)logger = logging.getLogger(__name__)



class Logger:class Logger:

    """    """

    Logger Service for MLflow model logging.    Logger Service for MLflow model logging.

    This class provides the log_model functionality for packaging RAG-based    This class provides the log_model functionality for packaging BERT-based

    conversational AI with document retrieval capabilities.    vacation recommendation models using the generic models-from-code pattern.

    """    """

        

    def __init__(self):    def __init__(self):

        """Initialize the logger service for logging purposes."""        """Initialize the logger service for logging purposes."""

        logger.info("Logger initialized for MLflow model logging")        logger.info("Logger initialized for MLflow model logging")



    @classmethod    @classmethod

    def log_model(    def log_model(

        cls,        cls,

        signature,        signature,

        artifact_path="AIStudio-Model",        artifact_path="AIStudio-Model",

        config_path="configs/config.yaml",        config_path="configs/config.yaml",

        docs_path="data/",        corpus_path=None,

        secrets_dict=None,        embeddings_path=None,

        model_path=None,        tokenizer_dir=None,

        demo_folder=None        bert_model_path=None,

    ):        demo_path=None,

        """        **kwargs

        Log model using refined models-from-code approach with elegant directory structure.    ):

                """

        This implementation uses MLflow's models-from-code approach exclusively with proper        Log model using refined models-from-code approach with elegant directory structure.

        temp directory naming to avoid redundant nesting while maintaining full MLflow 3.1.0 compatibility.        

                This implementation uses MLflow's models-from-code approach exclusively with proper

        Final MLflow structure achieved:        temp directory naming to avoid redundant nesting while maintaining full MLflow 3.1.0 compatibility.

        /artifacts/        

          └── data/                    # MLflow automatically created        Final MLflow structure achieved:

              ├── config.yaml          # Configuration        /artifacts/

              ├── data/                # Documents directory (PDFs, etc.)          └── data/                    # MLflow automatically created

              ├── demo/                # UI components                ├── config.yaml          # Configuration

              ├── models/              # Model files (optional)              ├── data/                # Documents directory (PDFs, etc.)

              └── secrets.yaml         # Secrets (optional)              ├── demo/                # UI components  

                      ├── models/              # Model files (optional)

        Args:              └── secrets.yaml         # Secrets (optional)

            signature: MLflow ModelSignature defining input/output schema for the model        

            artifact_path: Path to store the model artifacts        Args:

            config_path: Path to the configuration file            signature: MLflow ModelSignature defining input/output schema for the model

            docs_path: Path to the documents directory            artifact_path: Path to store the model artifacts

            secrets_dict: Dict with secrets to persist as YAML (optional)            config_path: Path to the configuration file

            model_path: Path to the model file (optional)            docs_path: Path to the documents directory

            demo_folder: Path to the demo folder (optional)            secrets_dict: Dict with secrets to persist as YAML (optional)

                        model_path: Path to the model file (optional)

        Returns:            demo_folder: Path to the demo folder (optional)

            None            

        """        Returns:

        import mlflow            None

        import tempfile        """

        import shutil        import mlflow

        import os        import tempfile

        import yaml        import shutil

                import os

        # Create temp directory        import yaml

        temp_base = tempfile.gettempdir()        

        temp_dir = os.path.join(temp_base, "model_artifacts")        # Create temp directory

                temp_base = tempfile.gettempdir()

        # Clean slate for deterministic results        temp_dir = os.path.join(temp_base, "model_artifacts")

        if os.path.exists(temp_dir):        

            shutil.rmtree(temp_dir)        # Clean slate for deterministic results

        os.makedirs(temp_dir)        if os.path.exists(temp_dir):

                    shutil.rmtree(temp_dir)

        try:        os.makedirs(temp_dir)

            logger.info(f"Organizing artifacts in temp directory: {temp_dir}")        

            # Organize temp directory for clean final structure        try:

            # MLflow will place this under /artifacts/data/ automatically            logger.info(f"Organizing artifacts in temp directory: {temp_dir}")

                        # Organize temp directory for clean final structure

            # ✅ Config at root -> /artifacts/data/config.yaml            # MLflow will place this under /artifacts/data/ automatically

            if not os.path.exists(config_path):            

                raise FileNotFoundError(f"Config file not found at: {config_path}")            # ✅ Config at root -> /artifacts/data/config.yaml

            shutil.copy2(config_path, os.path.join(temp_dir, "config.yaml"))            if not os.path.exists(config_path):

            logger.info(f"Copied config from {config_path} to temp directory")                raise FileNotFoundError(f"Config file not found at: {config_path}")

                        shutil.copy2(config_path, os.path.join(temp_dir, "config.yaml"))

            # ✅ Create data subdirectory -> /artifacts/data/data/            logger.info(f"Copied config from {config_path} to temp directory")

            data_temp_dir = os.path.join(temp_dir, "data")            

            os.makedirs(data_temp_dir, exist_ok=True)            # ✅ Create data subdirectory -> /artifacts/data/data/

                        data_temp_dir = os.path.join(temp_dir, "data")

            # Copy documents to data subdirectory            os.makedirs(data_temp_dir, exist_ok=True)

            if docs_path and os.path.exists(docs_path):            

                for item in os.listdir(docs_path):            # Copy documents to data subdirectory

                    item_path = os.path.join(docs_path, item)            if docs_path and os.path.exists(docs_path):

                    if os.path.isfile(item_path):                for item in os.listdir(docs_path):

                        shutil.copy2(item_path, data_temp_dir)                    item_path = os.path.join(docs_path, item)

                        logger.info(f"Copied document: {item}")                    if os.path.isfile(item_path):

                    elif os.path.isdir(item_path):                        shutil.copy2(item_path, data_temp_dir)

                        shutil.copytree(item_path, os.path.join(data_temp_dir, item))                        logger.info(f"Copied document: {item}")

                        logger.info(f"Copied document directory: {item}")                    elif os.path.isdir(item_path):

            else:                        shutil.copytree(item_path, os.path.join(data_temp_dir, item))

                logger.info("docs_path not provided or doesn't exist - skipping")                        logger.info(f"Copied document directory: {item}")

                        logger.info("data folder not provided or doesn't exist - skipping")

            # ✅ Demo folder -> /artifacts/data/demo/            

            if demo_folder and os.path.exists(demo_folder):            

                shutil.copytree(demo_folder, os.path.join(temp_dir, "demo"))            # ✅ Demo folder -> /artifacts/data/demo/

                logger.info(f"Copied demo folder from {demo_folder}")            if demo_folder and os.path.exists(demo_folder):

            else:                shutil.copytree(demo_folder, os.path.join(temp_dir, "demo"))

                logger.info("Demo folder not provided or doesn't exist - skipping")                logger.info(f"Copied demo folder from {demo_folder}")

                        else:

            # ✅ Handle secrets -> /artifacts/data/secrets.yaml                logger.info("Demo folder not provided or doesn't exist - skipping")

            if secrets_dict:            

                with open(os.path.join(temp_dir, "secrets.yaml"), 'w') as f:            # ✅ Handle secrets -> /artifacts/data/secrets.yaml

                    yaml.safe_dump(secrets_dict, f)            if secrets_dict:

                logger.info("Created secrets.yaml in temp directory")                with open(os.path.join(temp_dir, "secrets.yaml"), 'w') as f:

                                        yaml.safe_dump(secrets_dict, f)

            # ✅ Handle model files -> /artifacts/data/models/                logger.info("Created secrets.yaml in temp directory")

            if model_path and os.path.exists(model_path):                    

                models_temp_dir = os.path.join(temp_dir, "models")            # ✅ Handle model files -> /artifacts/data/models/

                os.makedirs(models_temp_dir, exist_ok=True)            if model_path and os.path.exists(model_path):

                if os.path.isfile(model_path):                models_temp_dir = os.path.join(temp_dir, "models")

                    shutil.copy2(model_path, os.path.join(models_temp_dir, os.path.basename(model_path)))                os.makedirs(models_temp_dir, exist_ok=True)

                    logger.info(f"Copied model file: {os.path.basename(model_path)}")                if os.path.isfile(model_path):

                else:                    shutil.copy2(model_path, os.path.join(models_temp_dir, os.path.basename(model_path)))

                    shutil.copytree(model_path, models_temp_dir, dirs_exist_ok=True)                    logger.info(f"Copied model file: {os.path.basename(model_path)}")

                    logger.info(f"Copied model directory: {model_path}")                else:

            else:                    shutil.copytree(model_path, models_temp_dir, dirs_exist_ok=True)

                logger.info("Model path not provided or doesn't exist - skipping")                    logger.info(f"Copied model directory: {model_path}")

                        else:

            mlflow.pyfunc.log_model(                logger.info("Model path not provided or doesn't exist - skipping")

                name=artifact_path,                                      

                loader_module="src.mlflow.loader",              mlflow.pyfunc.log_model(

                data_path=temp_dir,                                                   name=artifact_path,                          

                code_paths=["../src"],                                    loader_module="src.mlflow.loader",  

                signature=signature,                data_path=temp_dir,                                   

                pip_requirements="../requirements.txt"                code_paths=["../src"],                    

            )                signature=signature,

        except Exception as e:                pip_requirements="../requirements.txt"

            logger.error(f"Error during model logging: {str(e)}")            )

            raise        except Exception as e:

        finally:            logger.error(f"Error during model logging: {str(e)}")

            # Clean up temporary directory            raise

            if os.path.exists(temp_dir):        finally:

                shutil.rmtree(temp_dir)            # Clean up temporary directory

                logger.info("Cleaned up temporary directory")            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary directory")