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
        secrets_dict=None,
        model_path=None,
        demo_folder=None,
    ):
        """
        Log model using MLflow's models-from-code approach.

        This method includes model pre-caching to ensure deployment works offline.

        Final MLflow structure:
        /artifacts/
          └── data/
              ├── config.yaml
              ├── demo/
              ├── models/
              └── secrets.yaml

        Args:
            signature: MLflow ModelSignature defining input/output schema
            artifact_path: Path to store the model artifacts
            config_path: Path to the configuration file
            secrets_dict: Dict with secrets to persist as YAML (optional)
            model_path: Path to the model file (optional)
            demo_folder: Path to the demo folder (optional)
        """
        import mlflow
        import tempfile
        import shutil
        import os
        import yaml

        # Pre-cache models for deployment to ensure cache-only availability
        # Skip if using local datafabric models
        cls._ensure_models_cached(secrets_dict, config_path)

        # Create temp directory
        temp_base = tempfile.gettempdir()
        temp_dir = os.path.join(temp_base, "model_artifacts")

        # Clean slate for deterministic results
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        try:
            logger.info(f"Organizing artifacts in temp directory: {temp_dir}")

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at: {config_path}")
            shutil.copy2(config_path, os.path.join(temp_dir, "config.yaml"))
            logger.info(f"Copied config from {config_path}")

            if demo_folder and os.path.exists(demo_folder):
                shutil.copytree(demo_folder, os.path.join(temp_dir, "demo"))
                logger.info(f"Copied demo folder")

            if secrets_dict:
                with open(os.path.join(temp_dir, "secrets.yaml"), "w") as f:
                    yaml.safe_dump(secrets_dict, f)
                logger.info("Created secrets.yaml")

            # Copy datafabric models to artifacts if they exist
            if config_path and os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f) or {}

                qwen_local_path = config.get("qwen_model_path")
                clap_local_path = config.get("clap_model_path")

                if (
                    qwen_local_path
                    and os.path.exists(qwen_local_path)
                    and clap_local_path
                    and os.path.exists(clap_local_path)
                ):
                    models_temp_dir = os.path.join(temp_dir, "models")
                    os.makedirs(models_temp_dir, exist_ok=True)

                    # Copy Qwen model
                    qwen_dest = os.path.join(models_temp_dir, "qwen")
                    shutil.copytree(qwen_local_path, qwen_dest, dirs_exist_ok=True)
                    logger.info(
                        f"Copied Qwen model from {qwen_local_path} to artifacts"
                    )

                    # Copy CLAP model
                    clap_dest = os.path.join(models_temp_dir, "clap")
                    shutil.copytree(clap_local_path, clap_dest, dirs_exist_ok=True)
                    logger.info(
                        f"Copied CLAP model from {clap_local_path} to artifacts"
                    )

                    logger.info("✅ Datafabric models packaged with MLflow model")

            # Log model with models-from-code approach (no artifacts parameter)
            mlflow.pyfunc.log_model(
                name=artifact_path,
                loader_module="src.mlflow.loader",
                data_path=temp_dir,
                code_paths=["../src"],
                signature=signature,
                pip_requirements="../requirements.txt",
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
    def _ensure_models_cached(cls, secrets_dict=None, config_path=None):
        """
        Pre-cache models to ensure they're available for cache-only deployment.
        Only caches when using remote models - skips if local datafabric models exist.
        """
        # Check if we should use local datafabric models
        if config_path and os.path.exists(config_path):
            import yaml

            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}

            qwen_local_path = config.get("qwen_model_path")
            clap_local_path = config.get("clap_model_path")

            if (
                qwen_local_path
                and os.path.exists(qwen_local_path)
                and clap_local_path
                and os.path.exists(clap_local_path)
            ):
                logger.info(
                    "🏠 Local datafabric models detected - skipping pre-caching"
                )
                logger.info(f"  Qwen: {qwen_local_path}")
                logger.info(f"  CLAP: {clap_local_path}")
                return

        logger.info("🔄 Pre-caching models for cache-only deployment...")

        # Set up HuggingFace token if available
        if secrets_dict and "AIS_HUGGINGFACE_API_KEY" in secrets_dict:
            os.environ["HF_TOKEN"] = secrets_dict["AIS_HUGGINGFACE_API_KEY"]

        from transformers import (
            ClapProcessor,
            ClapModel,
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        try:
            # Pre-cache CLAP model
            clap_repo = "laion/clap-htsat-unfused"
            logger.info(f"📥 Caching CLAP model: {clap_repo}")
            ClapProcessor.from_pretrained(clap_repo)
            ClapModel.from_pretrained(clap_repo)

            # Pre-cache Qwen model
            qwen_repo = os.environ.get("AUDIO_LLM_ID", "Qwen/Qwen2.5-Omni-7B")
            logger.info(f"📥 Caching Qwen model: {qwen_repo}")
            Qwen2_5OmniProcessor.from_pretrained(qwen_repo, trust_remote_code=True)
            Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                qwen_repo,
                trust_remote_code=True,
                torch_dtype="auto",
            )

            logger.info("✅ Models successfully cached for deployment")

        except Exception as e:
            logger.error(f"❌ Failed to cache models: {e}")
            raise RuntimeError(f"Model caching failed - deployment will not work: {e}")
