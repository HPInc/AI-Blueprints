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
import time
from typing import Any, Dict, Optional
import pandas as pd

# Add the src directory to the path to import utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.prompt_templates import get_markdown_correction_prompt
from src.utils import initialize_llm
from src.parser import parse_md_for_grammar_correction, restore_placeholders
from src.chunker import chunk_markdown
from src.github_extractor import GitHubMarkdownProcessor
from src.llm_metrics import semantic_similarity_eval_fn, readability_improvement_eval_fn, llm_judge_eval_fn_local

# Set up logger
logger = logging.getLogger(__name__)


class Model:
    """
    Standalone model class with no MLflow inheritance.
    Handles grammar and structure correction in Markdown content.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        secrets: Dict[str, Any] = None,
        model_path: str = None,
    ):
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
            local_model_path=model_path or config.get("model_path"),
        )

        # Create the LLM chain
        self.llm_chain = self.prompt | self.llm

        logger.info("Grammar correction model initialized successfully.")

    def predict(
        self, model_input: pd.DataFrame, params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Applies the full grammar correction pipeline to markdown content.

        Args:
            model_input: DataFrame with columns:
                - repo_url: Optional GitHub repository URL
                - files: Markdown content to correct (either raw markdown or file paths)
            params: Optional parameters (unused but maintained for compatibility)

        Returns:
            DataFrame with columns:
                - corrected: Dict with corrected markdown files
                - originals: Dict with original markdown files
                - response_time: Processing time in seconds
                - evaluation_metrics: Dict with evaluation metrics

        Raises:
            KeyError: If input DataFrame is missing required columns
        """
        results = []
        
        for _, row in model_input.iterrows():
            start_time = time.time()
            
            # Get markdown content (from files column)
            markdown_content = row.get("files", "")
            
            # Parse markdown to protect structure
            placeholder_map, parsed_content = parse_md_for_grammar_correction(markdown_content)
            
            # Chunk the parsed content
            chunks = chunk_markdown(parsed_content, max_tokens=100)
            
            # Correct each chunk
            corrected_chunks = []
            for chunk in chunks:
                try:
                    corrected_chunk = self.llm_chain.invoke({"markdown": chunk})
                    corrected_chunks.append(corrected_chunk)
                except Exception as e:
                    logger.warning(f"Error correcting chunk: {e}")
                    corrected_chunks.append(chunk)
            
            # Reassemble chunks
            corrected_parsed = "\n".join(corrected_chunks)
            
            # Restore placeholders
            corrected_content = restore_placeholders(corrected_parsed, placeholder_map)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Calculate evaluation metrics
            try:
                semantic_sim = semantic_similarity_eval_fn([corrected_content], [markdown_content])
                readability = readability_improvement_eval_fn([corrected_content], [markdown_content])
                llm_judge = llm_judge_eval_fn_local(
                    pd.Series([corrected_content]), 
                    pd.Series([markdown_content]), 
                    self.llm
                )
                
                evaluation_metrics = {
                    "semantic_similarity": float(semantic_sim),
                    "readability_improvement": float(readability),
                    "llm_judge_score": float(llm_judge)
                }
            except Exception as e:
                logger.warning(f"Error calculating metrics: {e}")
                evaluation_metrics = {
                    "semantic_similarity": 0.0,
                    "readability_improvement": 0.0,
                    "llm_judge_score": 0.0
                }
            
            results.append({
                "corrected": {"corrected_file.md": corrected_content},
                "originals": {"original_file.md": markdown_content},
                "response_time": response_time,
                "evaluation_metrics": evaluation_metrics
            })
        
        return pd.DataFrame(results)
