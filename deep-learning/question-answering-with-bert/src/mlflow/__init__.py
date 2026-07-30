"""
MLflow models-from-code implementation for BERT Question Answering.

This module provides the standardized universal structure for MLflow model logging
and loading using the models-from-code approach.
"""

__all__ = ["Model", "Logger", "ExtractiveQAPipeline"]


def __getattr__(name):
    """Dynamic import for backwards compatibility and lazy loading."""
    if name == "Model":
        from .model import Model

        return Model
    if name == "Logger":
        from .logger import Logger

        return Logger
    if name == "ExtractiveQAPipeline":
        from .model import ExtractiveQAPipeline

        return ExtractiveQAPipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
