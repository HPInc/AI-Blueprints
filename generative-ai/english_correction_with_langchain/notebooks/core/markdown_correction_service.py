import mlflow
import pandas as pd
from typing import Any, Dict, List
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec

class MarkdownCorrectionService(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from src.utils import load_config_and_secrets, initialize_llm
        from core.prompt_templates import get_markdown_correction_prompt

        config_path = context.artifacts["config"]
        secrets_path = context.artifacts["secrets"]
        model_path = context.artifacts["llm"]

        config, secrets = load_config_and_secrets(config_path, secrets_path)
        self.prompt = get_markdown_correction_prompt()
        self.llm = initialize_llm(config["model_source"], secrets, model_path)
        self.llm_chain = self.prompt | self.llm

    def predict(self, context: Any, model_input: pd.DataFrame) -> list:
        corrected = []
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
    ):
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
                "textstat", # for ari_grade_level eval
            ],
            code_paths=["core", "../src"],
        )
