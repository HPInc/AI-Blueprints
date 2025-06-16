"""
Base service class for AI Studio Galileo Templates.
This module provides the core functionality for all service classes,
including model loading, configuration, and integration with Galileo services.
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
    """Base class for all generative services in AI Studio Galileo Templates."""

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
            "galileo_key": secrets.get("GALILEO_API_KEY", ""),
            "hf_key": secrets.get("HUGGINGFACE_API_KEY", ""),
            "galileo_url": config.get("galileo_url", "https://console.hp.galileocloud.io/"),
            "proxy": config.get("proxy", None),
            "model_source": config.get("model_source", "local"),
            "observe_project": f"{self.__class__.__name__}_Observations",
            "protect_project": f"{self.__class__.__name__}_Protection",
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
                    
            # Set up Galileo environment
            if not self.model_config.get("galileo_key"):
                logger.warning("No Galileo API key found. Galileo services will not function.")
            else:
                logger.info("Setting up Galileo environment variables.")
                os.environ["GALILEO_API_KEY"] = self.model_config["galileo_key"]
                os.environ["GALILEO_CONSOLE_URL"] = self.model_config["galileo_url"]
                
        except Exception as e:
            logger.error(f"Error setting up environment: {str(e)}")
            # Continue without failing to allow the model to still function even without Galileo
    
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
    
    def setup_protection(self) -> None:
        """Set up protection with Galileo Protect."""
        try:
            import galileo_protect as gp
            from galileo_protect import ProtectTool, ProtectParser, Ruleset
            
            logger.info(f"Setting up Galileo Protect for project: {self.model_config['protect_project']}")
            
            # Create project and stage
            project = gp.create_project(self.model_config["protect_project"])
            project_id = project.id
            
            timestamp = datetime.datetime.now()
            stage_name = f"{self.model_config['protect_project']}_stage" + timestamp.strftime('%Y-%m-%d %H:%M:%S')
            stage = gp.create_stage(name=stage_name, project_id=project_id)
            stage_id = stage.id
            
            # Create default ruleset for PII protection
            ruleset = Ruleset(
                rules=[
                    {
                        "metric": "pii",
                        "operator": "contains",
                        "target_value": "ssn",
                    },
                ],
                action={
                    "type": "OVERRIDE",
                    "choices": [
                        "Personal Identifiable Information detected in the model output. Sorry, I cannot answer that question."
                    ]
                }
            )
            
            # Create protection tool
            self.protect_tool = ProtectTool(
                stage_id=stage_id,
                prioritized_rulesets=[ruleset],
                timeout=10
            )
            
            # Set up protection parser and chain
            protect_parser = ProtectParser(chain=self.chain)
            self.protected_chain = self.protect_tool | protect_parser.parser
            logger.info("Galileo Protect setup successfully.")
        except Exception as e:
            logger.error(f"Failed to set up Galileo Protect: {str(e)}")
            # Fallback to unprotected chain if protection setup fails
            logger.warning("Using unprotected chain as fallback.")
            self.protected_chain = self.chain
    
    def setup_monitoring(self) -> None:
        """Set up monitoring with Galileo Observe."""
        try:
            from galileo_observe import GalileoObserveCallback
            
            logger.info(f"Setting up Galileo Observe for project: {self.model_config['observe_project']}")
            self.monitor_handler = GalileoObserveCallback(
                project_name=self.model_config["observe_project"]
            )
            logger.info("Galileo Observe setup successfully.")
        except Exception as e:
            logger.error(f"Failed to set up Galileo Observe: {str(e)}")
            # Create a dummy handler that does nothing when Galileo services aren't available
            self.monitor_handler = type('DummyHandler', (), {'on_llm_start': lambda *args, **kwargs: None, 
                                                              'on_llm_end': lambda *args, **kwargs: None,
                                                              'on_llm_error': lambda *args, **kwargs: None})()
    
    def setup_evaluation(self, scorers=None) -> None:
        """
        Set up evaluation with Galileo Prompt Quality.
        
        Args:
            scorers: List of scorer functions to use for evaluation
        """
        try:
            import promptquality as pq
            
            if scorers is None:
                scorers = [
                    pq.Scorers.context_adherence_luna,
                    pq.Scorers.correctness,
                    pq.Scorers.toxicity,
                    pq.Scorers.sexist
                ]
            
            logger.info(f"Setting up Galileo Evaluator for project: {self.model_config['observe_project']}")
            self.prompt_handler = pq.GalileoPromptCallback(
                project_name=self.model_config["observe_project"],
                scorers=scorers
            )
            logger.info("Galileo Evaluator setup successfully.")
        except Exception as e:
            logger.error(f"Failed to set up Galileo Evaluator: {str(e)}")
            # Create a dummy handler that does nothing when Galileo services aren't available
            self.prompt_handler = type('DummyHandler', (), {'on_llm_start': lambda *args, **kwargs: None, 
                                                             'on_llm_end': lambda *args, **kwargs: None,
                                                             'on_llm_error': lambda *args, **kwargs: None,
                                                             'finish': lambda *args, **kwargs: None})()
    
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
            
            # Set up Galileo integration with error handling
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