""""""

Logger Service implementation for MLflow model logging - COVID Movement Patterns with VAR.Logger Service implementation for MLflow model logging.



MLflow Registration LayerMLflow Registration Layer

- Provides log_model functionality for VAR models- Provides log_model functionality for models

- Handles artifact organization and temporary directory management- Handles artifact organization and temporary directory management

- Uses MLflow's models-from-code approach for deployment- Uses MLflow's models-from-code approach for deployment

- Manages configuration, trained models, and demo assets- Manages configuration, documents, secrets, and demo assets

""""""

import osimport os

import loggingimport uuid

import shutilimport base64

import tempfileimport logging

from typing import Dict, Any, Optionalimport shutil

import yamlfrom typing import Dict, Any, List

import yaml

# Set up loggerimport tempfile

logger = logging.getLogger(__name__)import pandas as pd



class Logger:# Set up logger

    """logger = logging.getLogger(__name__)

    Logger Service for MLflow model logging - COVID Movement Patterns with VAR.

    This class provides the log_model functionality for packaging VAR modelsclass Logger:

    for COVID-19 movement patterns forecasting.    """

    """    Logger Service for MLflow model logging.

        This class provides the log_model functionality for packaging RAG-based

    def __init__(self):    conversational AI with document retrieval capabilities.

        """Initialize the logger service for logging purposes."""    """

        logger.info("Logger initialized for MLflow model logging")    

    def __init__(self):

    @classmethod        """Initialize the logger service for logging purposes."""

    def log_model(        logger.info("Logger initialized for MLflow model logging")

        cls,

        signature,    @classmethod

        artifact_path="AIStudio-Model",    def log_model(

        config_path="configs/config.yaml",        cls,

        artifacts_dir="../artifacts",        signature,

        demo_folder=None        artifact_path="AIStudio-Model",

    ):        config_path="configs/config.yaml",

        """        docs_path="data/",

        Log model using models-from-code approach with VAR model artifacts.        secrets_dict=None,

                model_path=None,

        Final MLflow structure achieved:        demo_folder=None

        /artifacts/    ):

          └── data/                    # MLflow automatically created        """

              ├── config.yaml          # Configuration        Log model using refined models-from-code approach with elegant directory structure.

              ├── ny_model.pkl         # New York VAR model        

              ├── ldn_model.pkl        # London VAR model        This implementation uses MLflow's models-from-code approach exclusively with proper

              ├── ny_last_values.pkl   # NY last values for forecasting        temp directory naming to avoid redundant nesting while maintaining full MLflow 3.1.0 compatibility.

              ├── ldn_last_values.pkl  # London last values for forecasting        

              ├── ny_last_raw_value.pkl # NY last raw values        Final MLflow structure achieved:

              ├── ldn_last_raw_value.pkl # London last raw values        /artifacts/

              ├── features.pkl         # Feature names          └── data/                    # MLflow automatically created

              └── demo/                # UI components (optional)              ├── config.yaml          # Configuration

                      ├── data/                # Documents directory (PDFs, etc.)

        Args:              ├── demo/                # UI components  

            signature: MLflow ModelSignature defining input/output schema for the model              ├── models/              # Model files (optional)

            artifact_path: Path to store the model artifacts              └── secrets.yaml         # Secrets (optional)

            config_path: Path to the configuration file        

            artifacts_dir: Path to the artifacts directory containing trained models        Args:

            demo_folder: Path to the demo folder (optional)            signature: MLflow ModelSignature defining input/output schema for the model

                        artifact_path: Path to store the model artifacts

        Returns:            config_path: Path to the configuration file

            None            docs_path: Path to the documents directory

        """            secrets_dict: Dict with secrets to persist as YAML (optional)

        import mlflow            model_path: Path to the model file (optional)

                    demo_folder: Path to the demo folder (optional)

        # Create temp directory            

        temp_base = tempfile.gettempdir()        Returns:

        temp_dir = os.path.join(temp_base, "model_artifacts")            None

                """

        # Clean slate for deterministic results        import mlflow

        if os.path.exists(temp_dir):        import tempfile

            shutil.rmtree(temp_dir)        import shutil

        os.makedirs(temp_dir)        import os

                import yaml

        try:        

            logger.info(f"Organizing artifacts in temp directory: {temp_dir}")        # Create temp directory

                    temp_base = tempfile.gettempdir()

            # ✅ Config at root -> /artifacts/data/config.yaml        temp_dir = os.path.join(temp_base, "model_artifacts")

            if not os.path.exists(config_path):        

                raise FileNotFoundError(f"Config file not found at: {config_path}")        # Clean slate for deterministic results

            shutil.copy2(config_path, os.path.join(temp_dir, "config.yaml"))        if os.path.exists(temp_dir):

            logger.info(f"Copied config from {config_path} to temp directory")            shutil.rmtree(temp_dir)

                    os.makedirs(temp_dir)

            # ✅ Copy all model artifacts -> /artifacts/data/*.pkl        

            model_artifacts = [        try:

                "ny_model.pkl",            logger.info(f"Organizing artifacts in temp directory: {temp_dir}")

                "ldn_model.pkl",            # Organize temp directory for clean final structure

                "ny_last_values.pkl",            # MLflow will place this under /artifacts/data/ automatically

                "ldn_last_values.pkl",            

                "ny_last_raw_value.pkl",            # ✅ Config at root -> /artifacts/data/config.yaml

                "ldn_last_raw_value.pkl",            if not os.path.exists(config_path):

                "features.pkl"                raise FileNotFoundError(f"Config file not found at: {config_path}")

            ]            shutil.copy2(config_path, os.path.join(temp_dir, "config.yaml"))

                        logger.info(f"Copied config from {config_path} to temp directory")

            for artifact_file in model_artifacts:            

                artifact_source_path = os.path.join(artifacts_dir, artifact_file)            # ✅ Create data subdirectory -> /artifacts/data/data/

                if not os.path.exists(artifact_source_path):            data_temp_dir = os.path.join(temp_dir, "data")

                    raise FileNotFoundError(f"Required artifact not found: {artifact_source_path}")            os.makedirs(data_temp_dir, exist_ok=True)

                            

                artifact_dest_path = os.path.join(temp_dir, artifact_file)            # Copy documents to data subdirectory

                shutil.copy2(artifact_source_path, artifact_dest_path)            if docs_path and os.path.exists(docs_path):

                logger.info(f"Copied artifact: {artifact_file}")                for item in os.listdir(docs_path):

                                item_path = os.path.join(docs_path, item)

            # ✅ Demo folder -> /artifacts/data/demo/                    if os.path.isfile(item_path):

            if demo_folder and os.path.exists(demo_folder):                        shutil.copy2(item_path, data_temp_dir)

                shutil.copytree(demo_folder, os.path.join(temp_dir, "demo"))                        logger.info(f"Copied document: {item}")

                logger.info(f"Copied demo folder from {demo_folder}")                    elif os.path.isdir(item_path):

            else:                        shutil.copytree(item_path, os.path.join(data_temp_dir, item))

                logger.info("Demo folder not provided or doesn't exist - skipping")                        logger.info(f"Copied document directory: {item}")

                        logger.info("data folder not provided or doesn't exist - skipping")

            mlflow.pyfunc.log_model(            

                name=artifact_path,                                      

                loader_module="src.mlflow.loader",              # ✅ Demo folder -> /artifacts/data/demo/

                data_path=temp_dir,                                               if demo_folder and os.path.exists(demo_folder):

                code_paths=["../src"],                                    shutil.copytree(demo_folder, os.path.join(temp_dir, "demo"))

                signature=signature,                logger.info(f"Copied demo folder from {demo_folder}")

                pip_requirements="../requirements.txt"            else:

            )                logger.info("Demo folder not provided or doesn't exist - skipping")

            logger.info(f"Successfully logged model with artifact path: {artifact_path}")            

                        # ✅ Handle secrets -> /artifacts/data/secrets.yaml

        except Exception as e:            if secrets_dict:

            logger.error(f"Error during model logging: {str(e)}")                with open(os.path.join(temp_dir, "secrets.yaml"), 'w') as f:

            raise                    yaml.safe_dump(secrets_dict, f)

        finally:                logger.info("Created secrets.yaml in temp directory")

            # Clean up temp directory                    

            if os.path.exists(temp_dir):            # ✅ Handle model files -> /artifacts/data/models/

                shutil.rmtree(temp_dir)            if model_path and os.path.exists(model_path):

                logger.info("Temporary directory cleaned up")                models_temp_dir = os.path.join(temp_dir, "models")
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