import logging
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.types.schema import Schema, ColSpec
from mlflow.types import ParamSchema, ParamSpec
from mlflow.models import ModelSignature
from mlflow.pyfunc import PythonModel
from pathlib import Path
import sys
import os
from typing import Dict, Any

# Add the parent directory to sys.path to import modules
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from service.base_service import BaseGenerativeService
from utils import initialize_llm
from prompt_templates import format_summarization_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextSummarizationService(BaseGenerativeService):
    """Text summarization service using LLM and MLflow integration."""

    def load_model(self, context):
        """Load the text summarization model."""
        try:
            self.llm = initialize_llm(
                model_source=self.model_config["model_source"],
                secrets=self.model_config if "hf_key" in self.model_config else None,
                local_model_path=context.artifacts.get("model", None),
                hf_repo_id=""
            )
            logger.info("Text summarization model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def load_prompt(self):
        """Load the summarization prompt template."""
        try:
            self.prompt = format_summarization_prompt()
            logger.info("Summarization prompt template loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading prompt: {str(e)}")
            raise

    def load_chain(self):
        """Create the summarization processing chain."""
        try:
            from langchain.schema import StrOutputParser
            self.chain = self.prompt | self.llm | StrOutputParser()
            logger.info("Summarization chain created successfully.")
        except Exception as e:
            logger.error(f"Error creating chain: {str(e)}")
            raise

    def predict(self, context, model_input):
        """Generate text summaries."""
        try:
            if isinstance(model_input, dict):
                text_content = model_input.get("text", "")
                max_length = model_input.get("max_length", 150)
            else:
                text_content = str(model_input)
                max_length = 150

            if not text_content:
                return "Error: No text provided for summarization."

            # Generate summary
            result = self.chain.invoke({
                "text": text_content,
                "max_length": max_length
            })

            return result

        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            return f"Error: {str(e)}"

# MLflow model registration functions
def register_model(
    model_name: str = "text-summarization-model",
    config_path: str = "../../configs/config.yaml",
    secrets_path: str = "../../configs/secrets.yaml",
    model_path: str = None,
    description: str = "Text Summarization Model using LLM"
):
    """Register the text summarization model with MLflow."""
    
    try:
        # Define model signature
        input_schema = Schema([
            ColSpec("string", "text"),
            ColSpec("long", "max_length")
        ])
        
        output_schema = Schema([ColSpec("string")])
        
        signature = ModelSignature(
            inputs=input_schema,
            outputs=output_schema,
            params=ParamSchema([
                ParamSpec("max_length", "long", 150)
            ])
        )
        
        # Define artifacts
        artifacts = {
            "config": config_path,
            "secrets": secrets_path
        }
        
        if model_path:
            artifacts["model"] = model_path
        
        # Log model
        with mlflow.start_run() as run:
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=TextSummarizationService(),
                artifacts=artifacts,
                signature=signature,
                registered_model_name=model_name,
                pip_requirements=[
                    "langchain>=0.1.0",
                    "langchain-community>=0.0.20",
                    "langchain-huggingface>=0.0.1",
                    "transformers>=4.30.0",
                    "torch>=2.0.0",
                    "llama-cpp-python>=0.2.0",
                    "PyYAML>=6.0",
                    "webvtt-py>=0.5.1"
                ]
            )
            
        logger.info(f"Model '{model_name}' registered successfully with MLflow.")
        return run.info.run_id
        
    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")
        raise

def load_registered_model(model_name: str, version: str = "latest"):
    """Load a registered model from MLflow."""
    try:
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Model '{model_name}' version '{version}' loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading registered model: {str(e)}")
        raise

def test_model_prediction(model, sample_text: str = "This is a sample text for summarization testing."):
    """Test model prediction with sample data."""
    try:
        result = model.predict({
            "text": sample_text,
            "max_length": 50
        })
        logger.info(f"Test prediction successful. Result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error during test prediction: {str(e)}")
        raise
