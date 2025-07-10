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

# Add basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseGenerativeService(PythonModel):
    """Base class for all generative services in AI Studio Templates."""

    def __init__(self):
        """Initialize the base service with empty configuration."""
        self.model_config = {}
        self.llm = None
        self.chain = None
        self.prompt = None
        self.callback_manager = None

    def load_config(self, context) -> Dict[str, Any]:
        """
        Load configuration from context artifacts.
        
        Args:
            context: MLflow model context containing artifacts
            
        Returns:
            Dictionary containing the loaded configuration
        """
        config_path = context.artifacts["config"]
        secrets_path = context.artifacts["secrets"]
        
        # Load configuration
        if os.path.exists(config_path):
            with open(config_path) as file:
                config = yaml.safe_load(file)
                logger.info(f"Configuration loaded from {config_path}")
        else:
            config = {}
            logger.warning(f"Configuration file not found at {config_path}")
            
        # Load secrets
        if os.path.exists(secrets_path):
            with open(secrets_path) as file:
                secrets = yaml.safe_load(file)
                logger.info(f"Secrets loaded from {secrets_path}")
        else:
            secrets = {}
            logger.warning(f"Secrets file not found at {secrets_path}")
            
        # Merge configurations
        self.model_config = {
            "hf_key": secrets.get("HUGGINGFACE_API_KEY", ""),
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
                logger.info("No proxy configuration found. Checking system environment variables.")
                # Check if proxy is set in environment variables
                system_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
                if system_proxy:
                    logger.info(f"Using system proxy: {system_proxy}")
                else:
                    logger.warning("No proxy configuration found in config or environment variables.")
                    
        except Exception as e:
            logger.error(f"Error setting up environment: {str(e)}")
            # Continue without failing to allow the model to still function
    
    def load_model(self, context) -> None:
        """
        Load the appropriate model based on configuration.
        
        Args:
            context: MLflow model context containing artifacts
        """
        raise NotImplementedError("Each service must implement its own model loading logic")
    
    def load_prompt(self) -> None:
        """Load the prompt template for the service."""
        raise NotImplementedError("Each service must implement its own prompt loading logic")
    
    def load_chain(self) -> None:
        """Create the processing chain using the loaded model and prompt."""
        raise NotImplementedError("Each service must implement its own chain creation logic")
    
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
            
            logger.info(f"{self.__class__.__name__} successfully loaded and configured.")
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
        raise NotImplementedError("Each service must implement its own prediction logic")