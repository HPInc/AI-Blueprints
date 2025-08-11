"""
Simplified TextGenerationService for registration only.

Registration Layer
- Handles MLflow model registration using models-from-code approach
- NO business logic (moved to TextGenerationModel)
- NO MLflow inheritance - pure registration functionality
- Maintains identical external API behavior
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
import yaml
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

from .text_generation_loader import save_model
from .text_generation_model import TextGenerationModel


class TextGenerationService:
    """Simplified service for registration only. Business logic moved to Model."""

    def __init__(self, model_instance: Optional[TextGenerationModel] = None):
        """
        Optional model instance for registration purposes.
        
        Args:
            model_instance: Pre-configured TextGenerationModel instance
        """
        self.model = model_instance

    @classmethod
    def log_model(
        cls,
        *,
        artifact_path: str = "script_generation_model",
        llm_artifact: str = "models/",
        config_path: str = "configs/config.yaml",
        secrets_dict: Dict = None,
        demo_folder: str = None,
    ):
        """
        Register model using models-from-code approach.
        
        Args:
            artifact_path: Path to store the model artifacts
            llm_artifact: Path to the LLM artifact
            config_path: Path to the configuration file
            secrets_dict: Dict with secrets to persist as YAML (optional)
            demo_folder: Path to the demo folder (optional)
            
        Returns:
            None
        """
        # Import here to avoid circular imports
        from .text_generation_loader import _add_project_to_syspath
        
        core, src = _add_project_to_syspath()
        
        artifacts = {
            "config": str(Path(config_path).resolve()),
            "llm": llm_artifact,
        }

        if secrets_dict:
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
            yaml.safe_dump(secrets_dict, tmp)
            tmp.close()
            artifacts["secrets"] = tmp.name
            logging.info(f"Secrets artifact written to temporary file {tmp.name}")
        
        # Add demo folder to artifacts if provided
        if demo_folder:
            artifacts["demo"] = str(Path(demo_folder).resolve())
        
        # Create model signature
        signature = ModelSignature(
            inputs=Schema(
                [
                    ColSpec("string", "query"),
                    ColSpec("integer", "max_results"),
                    ColSpec("integer", "chunk_size"),
                    ColSpec("integer", "chunk_overlap"),
                    ColSpec("boolean", "do_extract"),
                    ColSpec("boolean", "do_analyze"),
                    ColSpec("boolean", "do_generate"),
                    ColSpec("string", "analysis_prompt"),
                    ColSpec("string", "generation_prompt"),
                ]
            ),
            outputs=Schema(
                [
                    ColSpec("string", "extracted_papers"),
                    ColSpec("string", "script"),
                ]
            ),
        )

        # Use models-from-code approach
        mlflow.models.save_model(
            path=artifact_path,
            loader_module="core.text_generation_service.text_generation_loader",
            data_path=None,
            signature=signature,
            conda_env=None,
            pip_requirements="../requirements.txt",
            code_paths=[str(core)] + ([str(src)] if src else []),
        )
