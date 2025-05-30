"""
Text Summarization Service implementation that extends the BaseGenerativeService.

This service provides text summarization capabilities using different LLM options
and integrates with Galileo for protection, observation, and evaluation.
"""

import os
import logging
from typing import Dict, Any, Union
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from galileo_protect import ProtectParser

# Import base service class from the shared location
import sys
import os

# Add the src directory to the path to import base_service
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.service.base_service import BaseGenerativeService

# Set up logger
logger = logging.getLogger(__name__)

class TextSummarizationService(BaseGenerativeService):
    """Text Summarization Service that extends the BaseGenerativeService."""

    def load_model(self, context) -> None:
        """
        Load the appropriate model based on configuration.
        
        Args:
            context: MLflow model context containing artifacts
        """
        try:
            model_source = self.model_config.get("model_source", "local")
            logger.info(f"Attempting to load model from source: {model_source}")
            
            if model_source == "local":
                logger.info("Using local LlamaCpp model source")
                self.load_local_model(context)
            elif model_source == "hugging-face-local":
                logger.info("Using local Hugging Face model source")
                self.load_local_hf_model(context)
            elif model_source == "hugging-face-cloud":
                logger.info("Using cloud Hugging Face model source")
                self.load_cloud_hf_model(context)
            else:
                logger.error(f"Unsupported model source: {model_source}")
                raise ValueError(f"Unsupported model source: {model_source}")
                
            if self.llm is None:
                logger.error("Model failed to initialize - llm is None after loading")
                raise RuntimeError("Model initialization failed - llm is None")
                
            logger.info(f"Model of type {type(self.llm).__name__} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def load_local_model(self, context):
        """
        Load a local LlamaCpp model.
        
        Args:
            context: MLflow model context containing artifacts
        """
        try:
            logger.info("Initializing local LlamaCpp model.")
            model_path = context.artifacts.get("model", None)
            
            logger.info(f"Model path: {model_path}")
            
            if not model_path or not os.path.exists(model_path):
                logger.error(f"Model file not found at: {model_path}")
                raise FileNotFoundError(f"The model file was not found at: {model_path}")
            
            logger.info(f"Model file exists. Size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
            
            logger.info("Setting up callback manager")
            self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            logger.info("Initializing LlamaCpp with the following parameters:")
            logger.info(f"  - Model Path: {model_path}")
            logger.info(f"  - n_gpu_layers: 30, n_batch: 512, n_ctx: 4096")
            logger.info(f"  - max_tokens: 1024, f16_kv: True, temperature: 0.2")
            
            try:
                self.llm = LlamaCpp(
                    model_path=model_path,
                    n_gpu_layers=30,
                    n_batch=512,
                    n_ctx=4096,
                    max_tokens=1024,
                    f16_kv=True,
                    callback_manager=self.callback_manager,
                    verbose=False,
                    stop=[],
                    streaming=False,
                    temperature=0.2,
                )
                logger.info("LlamaCpp model initialized successfully.")
            except Exception as model_error:
                logger.error(f"Failed to initialize LlamaCpp model: {str(model_error)}")
                logger.error(f"Exception type: {type(model_error).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
                
            logger.info("Using local LlamaCpp model for text summarization.")
        except Exception as e:
            logger.error(f"Error in load_local_model: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def load_local_hf_model(self, context):
        """
        Load a local Hugging Face model.
        
        Args:
            context: MLflow model context containing artifacts
        """
        try:
            logger.info("Loading local Hugging Face model")
            model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            logger.info(f"Using model_id: {model_id}")
            
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            logger.info("Loading model...")
            model = AutoModelForCausalLM.from_pretrained(model_id)
            
            logger.info("Creating pipeline...")
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, device=0)
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("Using the local Deep Seek model downloaded from HuggingFace.")
        except Exception as e:
            logger.error(f"Error in load_local_hf_model: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def load_cloud_hf_model(self, context):
        """
        Load a cloud-based Hugging Face model.
        
        Args:
            context: MLflow model context containing artifacts
        """
        try:
            logger.info("Loading cloud Hugging Face model")
            if "hf_key" not in self.model_config:
                logger.error("Missing HuggingFace API key in model_config")
                raise ValueError("Missing required configuration: hf_key")
                
            logger.info("Initializing HuggingFaceEndpoint with Mistral-7B model")
            self.llm = HuggingFaceEndpoint(
                huggingfacehub_api_token=self.model_config["hf_key"],
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            )
            logger.info("Using the cloud Mistral model on HuggingFace.")
        except Exception as e:
            logger.error(f"Error in load_cloud_hf_model: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def load_prompt(self) -> None:
        """Load the prompt template for text summarization."""
        self.prompt_str = '''
            The following text is an excerpt of a transcription:

            ### 
            {context} 
            ###

            Please, summarize this transcription, in a concise and comprehensive manner.
            '''
        self.prompt = ChatPromptTemplate.from_template(self.prompt_str)

    def load_chain(self) -> None:
        """Create the summarization chain using the loaded model and prompt."""
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def predict(self, context, model_input):
        """
        Generate a summary from the input text.
        
        Args:
            context: MLflow model context
            model_input: Input data for summarization, expecting a "text" field
            
        Returns:
            Dictionary with the summary in a "summary" field
        """
        text = model_input["text"][0]
        try:
            logger.info("Processing summarization request")
            # Run the input through the protection chain with monitoring
            result = self.protected_chain.invoke(
                {"context": text}, 
                config={"callbacks": [self.monitor_handler]}
            )
            logger.info("Successfully processed summarization request")
            
            if isinstance(result, dict) and "predictions" in result and len(result["predictions"]) > 0:
                if "summary" in result["predictions"][0]:
                    summary = result["predictions"][0]["summary"]
                    logger.info("Extracted summary from predictions array")
                else:
                    logger.warning("Found predictions array but no summary field")
                    summary = str(result)
            else:
                # Use the result directly if it's a string or other format
                summary = result
                
            logger.info(f"Summary extraction completed, type: {type(summary)}")
            
        except Exception as e:
            error_message = f"Error processing summarization request: {str(e)}"
            logger.error(error_message)
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            summary = error_message
        
        # Return the result as a DataFrame with a summary column
        return pd.DataFrame([{"summary": summary}])
        
    @classmethod
    def log_model(cls, artifact_path, secrets_path, config_path, model_path=None, demo_folder=None):
        """
        Log the model to MLflow.
        
        Args:
            artifact_path: Path to store the model artifacts
            secrets_path: Path to the secrets file
            config_path: Path to the configuration file
            model_path: Path to the model file (optional)
            demo_folder: Path to the demo folder (optional)
            
        Returns:
            None
        """
        import mlflow
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec
        
        # Create demo folder if specified and doesn't exist
        if demo_folder and not os.path.exists(demo_folder):
            os.makedirs(demo_folder, exist_ok=True)
            
        # Define model input/output schema
        input_schema = Schema([
            ColSpec("string", "text")
        ])
        output_schema = Schema([
            ColSpec("string", "summary")
        ])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        # Prepare artifacts
        artifacts = {
            "secrets": secrets_path,
            "config": config_path
        }
        
        if demo_folder:
            artifacts["demo"] = demo_folder
            
        if model_path:
            artifacts["model"] = model_path
        
        # Log model to MLflow
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=cls(),
            artifacts=artifacts,
            signature=signature,
            code_paths=["../core", "../../src"],
            pip_requirements=[
                "galileo-protect==0.15.1",
                "galileo-observe==1.13.2",
                "pyyaml",
                "pandas",
                "sentence-transformers",
                "langchain_core",
                "langchain_huggingface",
                "tokenizers>=0.13.0",
                "httpx>=0.24.0",
            ]
        )
        logger.info("Model and artifacts successfully registered in MLflow.")