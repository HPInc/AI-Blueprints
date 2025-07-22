import mlflow
import pandas as pd
from typing import Any, Dict, List
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec
from utils import load_config_and_secrets, initialize_llm
from prompt_templates import get_markdown_correction_prompt

class MarkdownCorrectionService(mlflow.pyfunc.PythonModel):
    """
    MLflow PythonModel for correcting grammar and structure in Markdown content.

    This class loads a language model pipeline (LLM chain) and applies it to
    input Markdown text to generate grammatically corrected output.
    """
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Loads configuration, secrets, and LLM chain from provided MLflow artifacts.

        Args:
            context (mlflow.pyfunc.PythonModelContext): MLflow context that provides artifact paths.
        """

        # Extract artifact paths from MLflow context
        config_path = context.artifacts["config"]
        secrets_path = context.artifacts["secrets"]
        model_path = context.artifacts["llm"]

        # Load configuration and secrets from disk
        config, secrets = load_config_and_secrets(config_path, secrets_path)

        # Build the LLM pipeline
        self.prompt = get_markdown_correction_prompt()
        self.llm = initialize_llm(config["model_source"], secrets, model_path)
        self.llm_chain = self.prompt | self.llm

    def predict(self, context: Any, model_input: pd.DataFrame) -> list:
        """
        Applies the correction pipeline to each row of the input dataframe.

        Args:
            context (Any): Unused prediction context.
            model_input (pd.DataFrame): DataFrame with a 'markdown' column.

        Returns:
            List: Corrected markdown text for each input row.
        """
        
        corrected = []

        # Apply the LLM correction chain row by row
        for _, row in model_input.iterrows():
            output = self.llm_chain.invoke({"markdown": row["markdown"]})
            corrected.append(output)
        return pd.Series(corrected)

    @classmethod
    def log_model(
        cls,
        artifact_path: str = "markdown_corrector",
        llm_artifact: str = "models/",
        config_yaml: str = "configs/config.yaml",
        secrets_yaml: str = "configs/secrets.yaml",
    ) -> None:
        """
        Logs the MarkdownCorrectionService model as an MLflow PyFunc model.

        Args:
            artifact_path (str): Destination path in MLflow model registry.
            llm_artifact (str): Path to the saved LLM model artifact.
            config_yaml (str): Path to the config YAML file.
            secrets_yaml (str): Path to the secrets YAML file.
        """
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=cls(),
            artifacts={
                "config": config_yaml,
                "secrets": secrets_yaml,
                "llm": llm_artifact,
            },
            signature=ModelSignature(
                inputs=Schema([ColSpec("string", "markdown")]),
                outputs=Schema([ColSpec("string", "corrected")]),
            ),
            pip_requirements=[
                "langchain",
                "PyYAML",
                "transformers",
                "mlflow",
                "torch",
                "pydantic",  # for langchain/transformers
            ],
            code_paths=["../src"],
        )
