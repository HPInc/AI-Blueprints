# src/markdown_correction_service.py

import mlflow
import pandas as pd
import logging
from typing import Any, List
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec
from utils import load_config_and_secrets, initialize_llm
from prompt_templates import get_markdown_correction_prompt

# Configure a logger for this module
logger = logging.getLogger(__name__)

class MarkdownCorrectionService(mlflow.pyfunc.PythonModel):
    """
    MLflow PythonModel for correcting grammar and structure in Markdown content.
    """
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Loads configuration, secrets, and the LLM chain from provided artifacts.
        """
        config_path = context.artifacts["config"]
        secrets_path = context.artifacts["secrets"]
        model_path = context.artifacts["llm"]

        config, secrets = load_config_and_secrets(config_path, secrets_path)
        self.prompt = get_markdown_correction_prompt()
        self.llm = initialize_llm(config["model_source"], secrets, model_path)
        self.llm_chain = self.prompt | self.llm
        logger.info("MarkdownCorrectionService context loaded successfully.")

    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.Series:
        """
        Applies the correction pipeline to each row of the input dataframe.
        """
        # Ensure the input DataFrame has the 'markdown' column
        if "markdown" not in model_input.columns:
            raise KeyError("Input DataFrame is missing the required 'markdown' column.")
            
        corrected = []
        for _, row in model_input.iterrows():
            output = self.llm_chain.invoke({"markdown": row["markdown"]})
            corrected.append(output)
        return pd.Series(corrected, name="corrected")

    @classmethod
    def log_model(
        cls,
        model_name: str,
        llm_artifact_path: str,
        config_path: str,
        secrets_path: str,
        requirements_path: str,
        code_paths: List[str]
    ) -> None:
        """
        Logs and registers the MarkdownCorrectionService model in MLflow.

        Args:
            model_name (str): The name for the model in the registry.
            llm_artifact_path (str): Path to the local LLM model artifact (e.g., GGUF file).
            config_path (str): Path to the configuration YAML file.
            secrets_path (str): Path to the secrets YAML file.
            requirements_path (str): Path to the pip requirements.txt file.
            code_paths (List[str]): List of local source code paths to include.
        """
        artifacts = {
            "config": config_path,
            "secrets": secrets_path,
            "llm": llm_artifact_path,
        }
        
        signature = ModelSignature(
            inputs=Schema([ColSpec("string", "markdown")]),
            outputs=Schema([ColSpec("string", "corrected")]),
        )

        mlflow.pyfunc.log_model(
            artifact_path=model_name,
            python_model=cls(),
            artifacts=artifacts,
            signature=signature,
            registered_model_name=model_name, # Log and register in one step
            pip_requirements=requirements_path,
            code_paths=code_paths,
        )
        logger.info(f"Model '{model_name}' logged and registered successfully.")