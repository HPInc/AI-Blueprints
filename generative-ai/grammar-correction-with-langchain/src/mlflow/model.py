"""
Standalone Model class for Grammar Correction.

Business Logic Layer
- Handles grammar and structure correction in Markdown content
- Manages LLM initialization and chain processing
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

import os
import sys
import logging
from typing import Any, Dict, Optional
import pandas as pd

# Add the src directory to the path to import utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.prompt_templates import get_markdown_correction_prompt
from src.utils import initialize_llm

# Set up logger
logger = logging.getLogger(__name__)


class Model:
    """
    Standalone model class with no MLflow inheritance.
    Handles grammar and structure correction in Markdown content.
    """
    
    def __init__(self, config: Dict[str, Any], secrets: Dict[str, Any] = None, model_path: str = None):
        """
        Initialize the grammar correction model with configuration and secrets.
        
        Args:
            config: Configuration dictionary containing model settings
            secrets: Secrets dictionary containing API keys if needed
            model_path: Path to local model file (if using local model)
        """
        self.config = config
        self.secrets = secrets or {}
        self.model_path = model_path
        
        # Initialize the prompt template
        self.prompt = get_markdown_correction_prompt()
        
        # Initialize the LLM based on configuration
        self.llm = initialize_llm(
            model_source=config.get("model_source", "local"),
            secrets=secrets,
            local_model_path=model_path or config.get("model_path")
        )
        
        # Create the LLM chain
        self.llm_chain = self.prompt | self.llm
        
        logger.info("Grammar correction model initialized successfully.")

    def predict(self, model_input: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        """
        Applies the grammar correction pipeline to each row of the input dataframe.
        
        Args:
            model_input: DataFrame containing 'markdown' column with text to correct
            params: Optional parameters (unused but maintained for compatibility)
            
        Returns:
            Series containing corrected markdown text
            
        Raises:
            KeyError: If input DataFrame is missing the required 'markdown' column
        """
        # Ensure the input DataFrame has the 'markdown' column
        if "markdown" not in model_input.columns:
            raise KeyError("Input DataFrame is missing the required 'markdown' column.")

        corrected = []
        for _, row in model_input.iterrows():
            output = self.llm_chain.invoke({"markdown": row["markdown"]})
            corrected.append(output)
            
        return pd.Series(corrected, name="corrected")