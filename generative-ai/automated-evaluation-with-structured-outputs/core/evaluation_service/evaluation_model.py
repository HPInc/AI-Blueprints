"""
Standalone EvaluationModel class.

Business Logic Layer
- Handles automated evaluation of texts using LLaMA model with structured outputs
- Manages LLM initialization, scoring criteria, and prediction logic  
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

import re
import json
import multiprocessing
import logging
from typing import Dict, Any, Optional
import pandas as pd
from llama_cpp import Llama

# Set up logger
logger = logging.getLogger(__name__)


class EvaluationModel:
    """
    Standalone evaluation model class with no MLflow inheritance.
    Handles automated evaluation of texts using LLaMA model with structured outputs.
    """
    
    def __init__(self, llm_model_path: str, config: dict = None):
        """
        Initialize the EvaluationModel with LLM and configuration.
        
        Args:
            llm_model_path: Path to the LLaMA model file
            config: Model configuration dictionary (optional)
        """
        self.llm_model_path = llm_model_path
        self.config = config or {}
        self.llm = None
        
        # Initialize LLM
        try:
            self._load_llm()
            logger.info("EvaluationModel initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EvaluationModel: {str(e)}")
            raise RuntimeError(f"EvaluationModel initialization failed: {str(e)}") from e
    
    def _load_llm(self) -> None:
        """Load LLaMA model with optimized configuration."""
        try:
            self.llm = Llama(
                model_path=self.llm_model_path,
                n_gpu_layers=-1,
                n_batch=128,
                n_ctx=8192,
                max_tokens=512,
                f16_kv=True,
                use_mmap=False,
                low_vram=True,
                rope_scaling=None,
                temperature=0.0,
                repeat_penalty=1.0,
                streaming=False,
                stop=None,
                seed=42,
                num_threads=multiprocessing.cpu_count(),
                verbose=False,
            )
            logger.info(f"LLM loaded successfully from {self.llm_model_path}")
        except Exception as e:
            logger.error(f"Failed to load LLM: {str(e)}")
            raise

    def predict(self, model_input: pd.DataFrame, params: dict = None) -> pd.DataFrame:
        """
        Evaluate texts using LLaMA model and return scores with total.
        
        Args:
            model_input: DataFrame containing texts to evaluate
            params: Dictionary containing:
                - key_column: Column name for unique identifiers (default: "title")
                - eval_column: Column name for text to evaluate (default: "abstract") 
                - criteria: Dictionary mapping criteria names to max scores
                
        Returns:
            pd.DataFrame: Original data merged with evaluation scores and TotalScore
        """
        if params is None:
            params = {}
        
        # Default parameters
        key_col = params.get("key_column", "title")
        eval_col = params.get("eval_column", "abstract")
        criteria = params.get("criteria", {
            "Originality": 20,
            "Clarity": 20,
            "Relevance": 20,
            "Feasibility": 20,
            "Impact": 20
        })
        
        # Handle criteria as JSON string
        if isinstance(criteria, str):
            criteria = json.loads(criteria)

        # Validate input
        for col in (key_col, eval_col):
            if col not in model_input.columns:
                raise KeyError(f"Input DataFrame missing column '{col}'")

        df = model_input.copy()
        df[key_col] = df[key_col].astype(str)

        # Helper functions
        def scale_score(raw: int, target: int) -> int:
            """Scale raw score (1-10) to target range."""
            scaled = round((raw / 10) * target)
            return min(max(scaled, 0), target)

        def extract_score(text: str) -> int:
            """Extract numeric score from LLM response text."""
            match = re.search(r"\\b(10|[1-9])\\b", text)
            return int(match.group(1)) if match else -1

        def eval_criterion(text: str, crit: str) -> int:
            """Evaluate text against criterion using LLM."""
            prompt = (
                f"Evaluate abstract by '{crit}', return integer 1-10 only.\\n"
                f"Abstract:\\n{text.strip()}\\nScore:"
            )
            resp = self.llm(prompt)["choices"][0]["text"]
            return extract_score(resp)

        # Process each row
        results = []
        for _, row in df.iterrows():
            scores = {crit: scale_score(eval_criterion(row[eval_col], crit), criteria[crit])
                      for crit in criteria}
            scores[key_col] = row[key_col]
            results.append(scores)

        scored_df = pd.DataFrame(results)
        # Merge & compute total
        merged = df.merge(scored_df, on=key_col)
        merged["TotalScore"] = merged[list(criteria)].sum(axis=1)
        
        return merged
