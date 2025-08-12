"""
Agentic Service Package

This package provides a models-from-code implementation for agentic feedback analysis.
Includes model, loader, and service components for clean MLflow integration.
"""

from .agentic_model import AgenticModel, AgenticModelInput, AgenticModelOutput
from .agentic_service import AgenticService

__all__ = [
    "AgenticModel",
    "AgenticModelInput", 
    "AgenticModelOutput",
    "AgenticService",
]
