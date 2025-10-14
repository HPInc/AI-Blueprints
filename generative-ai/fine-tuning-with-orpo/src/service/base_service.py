"""
Base service class for AI Studio Templates.
This module provides the core functionality for all service classes,
including model loading and configuration.
"""

import datetime
import os
import yaml
import sys
import logging
from typing import Dict, Any, Optional, List, Union
import mlflow
from mlflow.pyfunc import PythonModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from src.utils import load_secrets_to_env

# Add basic logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaseGenerativeService(PythonModel):
    """Base class for all generative services in AI Studio Templates."""

    def __init__(self):
        """Initialize the base service with empty configuration."""
        self.model_config = {}
        self.llm = None
        self.chain = None
        self.protected_chain = None
        self.prompt = None
        self.callback_manager = None
        self.monitor_handler = None
        self.prompt_handler = None
        self.protect_tool = None

    def load_config(self, context) -> Dict[str, Any]:
        """
        Load configuration from context artifacts.

        Args:
            context: MLflow model context containing artifacts

        Returns:
            Dictionary containing the loaded configuration
        """
        config_path = context.artifacts["config"]

        # Load secrets into environment
        secrets_path = context.artifacts.get("secrets")
        if secrets_path and os.path.exists(secrets_path):
            try:
                load_secrets_to_env(secrets_path)
                logger.info(f"Secrets loaded from {secrets_path} into environment")
            except Exception as e:
                logger.warning(
                    f"Failed to load secrets artifact from {secrets_path} into environment: {e}"
                )

        # Retrieve the token from the current environment
        token = os.getenv("AIS_HUGGINGFACE_API_KEY", "")
        if not token.strip():
            logger.warning("Key AIS_HUGGINGFACE_API_KEY not found or empty")
        else:
            logger.info("Hugging Face token is available and loaded from environment")

        # Load configuration
        if os.path.exists(config_path):
            with open(config_path) as file:
                config = yaml.safe_load(file)
                logger.info(f"Configuration loaded from {config_path}")
        else:
            config = {}
            logger.warning(f"Configuration file not found at {config_path}")

        # Merge configurations
        self.model_config = {
            "hf_key": token,
            "proxy": config.get("proxy", None),
            "model_source": config.get("model_source", "local"),
        }

        return self.model_config

    def setup_environment(self) -> None:
        """Configure environment variables based on loaded configuration."""
        try:
            # Configure proxy if specified in config
            if "proxy" in self.model_config and self.model_config["proxy"]:
                logger.info(f"Setting up proxy: {self.model_config['proxy']}")
                os.environ["HTTPS_PROXY"] = self.model_config["proxy"]
                os.environ["HTTP_PROXY"] = self.model_config["proxy"]
            else:
                logger.info(
                    "No proxy configuration found. Checking system environment variables."
                )
                # Check if proxy is set in environment variables
                system_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get(
                    "HTTP_PROXY"
                )
                if system_proxy:
                    logger.info(f"Using system proxy: {system_proxy}")
                else:
                    logger.warning(
                        "No proxy configuration found in config or environment variables."
                    )

            # Set up model environment variables
            if not self.model_config.get("hf_key"):
                logger.warning(
                    "No HuggingFace API key found. Some models may not function properly."
                )
            else:
                logger.info("Setting up HuggingFace environment variables.")
                os.environ["HUGGINGFACE_API_KEY"] = self.model_config["hf_key"]

        except Exception as e:
            logger.error(f"Error setting up environment: {str(e)}")
            # Continue without failing to allow the model to still function

    def load_model(self, context) -> None:
        """
        Load the appropriate model based on configuration.

        Args:
            context: MLflow model context containing artifacts
        """
        raise NotImplementedError(
            "Each service must implement its own model loading logic"
        )

    def load_prompt(self) -> None:
        """Load the prompt template for the service."""
        raise NotImplementedError(
            "Each service must implement its own prompt loading logic"
        )

    def load_chain(self) -> None:
        """Create the processing chain using the loaded model and prompt."""
        raise NotImplementedError(
            "Each service must implement its own chain creation logic"
        )

    def setup_protection(self) -> None:
        """Set up model output protection."""
        # This is a stub method that used to set up Galileo Protect
        # For now, we'll just use the unprotected chain
        logger.info("Using standard chain without additional protection.")
        self.protected_chain = self.chain

    def setup_monitoring(self) -> None:
        """Set up model usage monitoring."""
        # This is a stub method that used to set up monitoring
        logger.info("Monitoring functionality is not available.")
        # Create a dummy handler that does nothing
        self.monitor_handler = type(
            "DummyHandler",
            (),
            {
                "on_llm_start": lambda *args, **kwargs: None,
                "on_llm_end": lambda *args, **kwargs: None,
                "on_llm_error": lambda *args, **kwargs: None,
            },
        )()

    def setup_evaluation(self, scorers=None) -> None:
        """
        Set up evaluation for model outputs.

        Args:
            scorers: List of scorer functions to use for evaluation
        """
        logger.info("Evaluation functionality is not available.")
        # Create a dummy handler that does nothing
        self.prompt_handler = type(
            "DummyHandler",
            (),
            {
                "on_llm_start": lambda *args, **kwargs: None,
                "on_llm_end": lambda *args, **kwargs: None,
                "on_llm_error": lambda *args, **kwargs: None,
                "finish": lambda *args, **kwargs: None,
            },
        )()

    def load_context(self, context) -> None:
        """
        Load context for the model, including configuration, model, and chains.

        Args:
            context: MLflow model context
        """
        try:
            # Load configuration
            self.load_config(context)

            # Set up environment
            self.setup_environment()

            # Load model, prompt, and chain
            self.load_model(context)
            self.load_prompt()
            self.load_chain()

            # Set up protection, monitoring and evaluation with error handling
            try:
                self.setup_protection()
            except Exception as e:
                logger.error(f"Error setting up protection: {str(e)}")
                self.protected_chain = self.chain  # Fallback to unprotected chain

            try:
                self.setup_monitoring()
            except Exception as e:
                logger.error(f"Error setting up monitoring: {str(e)}")

            try:
                self.setup_evaluation()
            except Exception as e:
                logger.error(f"Error setting up evaluation: {str(e)}")

            logger.info(
                f"{self.__class__.__name__} successfully loaded and configured."
            )
        except Exception as e:
            logger.error(f"Error loading context: {str(e)}")
            raise

    def predict(self, context, model_input):
        """
        Make predictions using the loaded model.

        Args:
            context: MLflow model context
            model_input: Input data for prediction

        Returns:
            Model predictions
        """
        raise NotImplementedError(
            "Each service must implement its own prediction logic"
        )
