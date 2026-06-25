"""
Standalone Model class.

Business Logic Layer
- Handles text summarization using different LLM options
- Manages model initialization, prompt templates, and prediction logic
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

import os
import logging
from typing import Dict, Any, Union, List
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langchain_community.llms import LlamaCpp

# Fix for Pydantic model rebuild issue
if hasattr(LlamaCpp, "model_rebuild"):
    LlamaCpp.model_rebuild()
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set up logger
logger = logging.getLogger(__name__)


class Model:
    """
    Standalone model class containing all text summarization business logic.
    NO MLflow inheritance - pure domain functionality.
    """

    def __init__(self, llm, config: Dict[str, Any], prompt_str: str = None):
        """
        Direct dependency injection - no MLflow context.
        Extract all initialization logic from original service.

        Args:
            llm: Initialized language model instance
            config: Configuration dictionary containing model settings
            prompt_str: Formatted prompt template string (optional)
        """
        self.llm = llm
        self.config = config
        self.prompt_str = prompt_str
        self.prompt = None
        self.chain = None

        # Initialize prompt and chain if prompt_str provided
        if prompt_str:
            self.setup_prompt_and_chain(prompt_str)

        logger.info(f"Model initialized with {type(self.llm).__name__} model")

    def setup_prompt_and_chain(self, prompt_str: str):
        """
        Setup the prompt and chain for text summarization.

        Args:
            prompt_str: Formatted prompt template string
        """
        self.prompt_str = prompt_str
        self.prompt = ChatPromptTemplate.from_template(prompt_str)
        self.chain = self.prompt | self.llm | StrOutputParser()
        logger.info("Prompt and chain initialized successfully")

    def predict(self, model_input, params=None) -> pd.DataFrame:
        """
        Core business logic extracted from original service predict method.

        UI sends: {"inputs": {"text": [content]}, "params": {}}
        MLflow converts to: DataFrame with 'text' column containing [content] array

        Args:
            model_input: pandas DataFrame with 'text' column containing array             params: Optional prediction parameters (not used in current implementation

        Returns:
            DataFrame with the summary in a "summary" field
        """
        try:
            logger.info("Processing summarization request")
            logger.info(f"Received model_input type: {type(model_input)}")

            # MLflow signature guarantees DataFrame format with 'text' column
            if not hasattr(model_input, "iloc"):
                raise ValueError(f"Expected pandas DataFrame, got {type(model_input)}")

            if len(model_input) == 0:
                raise ValueError("Empty DataFrame received")

            if "text" not in model_input.columns:
                raise ValueError(
                    f"Expected 'text' column, got columns: {list(model_input.columns)}"
                )

            # Extract text from DataFrame
            text = model_input["text"][0]

            logger.info(f"Extracted text length: {len(text)}")
            logger.info(f"Text preview: {text[:100]}...")

            # Ensure chain is available
            if not self.chain:
                raise RuntimeError(
                    "Chain not initialized. Call setup_prompt_and_chain first."
                )

            # Run the input through the summarization chain
            result = self.chain.invoke({"context": text})

            logger.info("Successfully processed summarization request")

            # Handle different result formats based on what the chain returns
            if (
                isinstance(result, dict)
                and "predictions" in result
                and len(result["predictions"]) > 0
            ):
                if "summary" in result["predictions"][0]:
                    summary = result["predictions"][0]["summary"]
                    logger.info("Extracted summary from predictions array")
                else:
                    summary = str(result)
            else:
                # Use the result directly if it's a string or other format
                summary = str(result)

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

    @staticmethod
    def create_local_llama_model(model_path: str, callback_manager=None):
        """
        Static factory method to create a local LlamaCpp model.

        Args:
            model_path: Path to the LlamaCpp model file
            callback_manager: Optional callback manager

        Returns:
            Initialized LlamaCpp model instance
        """
        try:
            logger.info("Initializing local LlamaCpp model.")
            logger.info(f"Model path: {model_path}")

            if not model_path or not os.path.exists(model_path):
                logger.error(f"Model file not found at: {model_path}")
                raise FileNotFoundError(
                    f"The model file was not found at: {model_path}"
                )

            logger.info(
                f"Model file exists. Size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB"
            )

            if not callback_manager:
                logger.info("Setting up callback manager")
                callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

            logger.info("Initializing LlamaCpp with parameters")

            llm = LlamaCpp(
                model_path=model_path,
                n_gpu_layers=30,
                n_batch=512,
                n_ctx=4096,
                max_tokens=1024,
                f16_kv=True,
                callback_manager=callback_manager,
                verbose=False,
                stop=[],
                streaming=False,
                temperature=0.2,
            )
            logger.info("LlamaCpp model initialized successfully.")
            return llm

        except Exception as e:
            logger.error(f"Error creating local LlamaCpp model: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    @staticmethod
    def create_local_hf_model(hf_token: str = None):
        """
        Static factory method to create a local Hugging Face model.

        Args:
            hf_token: Hugging Face API token (optional)

        Returns:
            Initialized HuggingFacePipeline model instance
        """
        try:
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token

            logger.info("Loading local Hugging Face model")
            model_id = "meta-llama/Llama-3.2-3B-Instruct"
            logger.info(f"Using model_id: {model_id}")

            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            logger.info("Loading model...")
            model = AutoModelForCausalLM.from_pretrained(model_id)

            logger.info("Creating pipeline...")
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=100,
                device=0,
            )

            llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("Local Hugging Face model initialized successfully.")
            return llm

        except Exception as e:
            logger.error(f"Error creating local HuggingFace model: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    @staticmethod
    def create_cloud_hf_model(hf_token: str):
        """
        Static factory method to create a cloud-based Hugging Face model.

        Args:
            hf_token: Hugging Face API token (required)

        Returns:
            Initialized HuggingFaceEndpoint model instance
        """
        try:
            logger.info("Loading cloud Hugging Face model")
            if not hf_token or not hf_token.strip():
                logger.error("Missing HuggingFace API key")
                raise ValueError("Missing required configuration: hf_token")

            logger.info("Initializing HuggingFaceEndpoint with Mistral-7B model")
            llm = HuggingFaceEndpoint(
                huggingfacehub_api_token=hf_token,
                repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            )
            logger.info("Cloud Hugging Face model initialized successfully.")
            return llm

        except Exception as e:
            logger.error(f"Error creating cloud HuggingFace model: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
