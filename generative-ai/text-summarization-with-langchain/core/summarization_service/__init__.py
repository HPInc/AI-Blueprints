"""
Text Summarization Service module.

This module provides text summarization capabilities using the models-from-code approach.
It includes the SummarizationModel for business logic, SummarizationService for registration,
and the loader module for MLflow integration.
"""

from .summarization_model import SummarizationModel
from .summarization_service import SummarizationService

__all__ = ["SummarizationModel", "SummarizationService"]
