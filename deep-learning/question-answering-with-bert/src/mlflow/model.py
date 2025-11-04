"""
Standalone Model class for BERT Question Answering.

Business Logic Layer
- Handles BERT-based question answering using transformers pipeline
- Manages model initialization and prediction logic
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

import logging
import os
from typing import Dict, Any, Optional
import torch
from transformers import pipeline

# Set up logger
logger = logging.getLogger(__name__)


class Model:
    """
    Standalone model class with no MLflow inheritance.
    Handles BERT-based question answering using transformers pipeline.
    """

    def __init__(
        self, model_checkpoint: str = "distilbert-base-cased", config: dict = None
    ):
        """
        Initialize the Model with configuration.

        Args:
            model_checkpoint: The HuggingFace model checkpoint to use
            config: Model configuration dictionary (optional)
        """
        self.model_checkpoint = model_checkpoint
        self.config = config or {}
        self.model = None

        # Initialize the question-answering pipeline
        try:
            self._load_model()
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}") from e

    def _load_model(self) -> None:
        """Load the question-answering pipeline using the specified model checkpoint."""
        try:
            # Initialize the pipeline with the model checkpoint
            self.model = pipeline(
                "question-answering",
                model=self.model_checkpoint,
                device=(
                    0 if torch.cuda.is_available() else -1
                ),  # GPU if available, otherwise CPU
            )
            logger.info(
                f"Question-answering pipeline loaded successfully with model: {self.model_checkpoint}"
            )
        except Exception as e:
            logger.error(f"Error loading the question-answering pipeline: {str(e)}")
            # Try loading without device specification for compatibility
            try:
                self.model = pipeline("question-answering", model=self.model_checkpoint)
                logger.info(
                    "Question-answering pipeline loaded successfully (CPU fallback)"
                )
            except Exception as fallback_error:
                logger.error(
                    f"Failed to load pipeline even with fallback: {str(fallback_error)}"
                )
                raise

    def _preprocess(self, inputs: Dict[str, Any]) -> tuple:
        """
        Preprocesses the input data.

        Args:
            inputs: A dictionary containing two keys:
                - 'context': A list with the context text.
                - 'question': A list with the question to be answered.

        Returns:
            tuple: A tuple containing the context (str) and the question (str).
        """
        try:
            context = inputs["context"][0]
            question = inputs["question"][0]
            logger.info(
                f"Preprocessing - Context: {context[:50]}..., Question: {question}"
            )
            return context, question
        except Exception as e:
            logger.error(f"Error preprocessing the input data: {str(e)}")
            raise

    def predict(
        self, model_input: Dict[str, Any], params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Runs inference using the loaded model and input data.

        Args:
            model_input: A dictionary containing 'context' and 'question' keys.
            params: Optional parameters for inference.

        Returns:
            The output from the model containing the predicted answer and optionally the score.
        """
        try:
            # Check if model is loaded
            if self.model is None:
                raise ValueError(
                    "Model not loaded. Please ensure initialization was successful."
                )

            context, question = self._preprocess(model_input)
            output = self.model(context=context, question=question)

            # Handle params if provided
            if params and params.get("show_score", False):
                return output
            else:
                # Return only essential fields for backward compatibility
                return {
                    "answer": output.get("answer", ""),
                    "score": output.get("score", 0.0),
                    "start": output.get("start", 0),
                    "end": output.get("end", 0),
                }
        except Exception as e:
            logger.error(f"Error running inference: {str(e)}")
            raise
