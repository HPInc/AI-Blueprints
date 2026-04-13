"""
MLflow models-from-code implementation for Vanilla RAG with LangChain.

This module provides the standardized universal structure for MLflow model logging
and loading using the models-from-code approach.
"""

__all__ = ["Model", "Logger", "CharModel"]


def __getattr__(name):
    """Dynamic import for backwards compatibility and lazy loading."""
    if name == "Model":
        from .model import Model

        return Model
    if name == "Logger":
        from .logger import Logger

        return Logger
    if name == "CharModel":
        from .model import CharModel

        return CharModel
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
