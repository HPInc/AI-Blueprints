"""
MLflow models-from-code implementation for Iris Classification with SVM.

This module provides the standardized universal structure for MLflow model logging
and loading using the models-from-code approach for iris flower classification.
"""

__all__ = ["Model", "Logger"]


def __getattr__(name):
    """Dynamic import for backwards compatibility and lazy loading."""
    if name == "Model":
        from .model import Model

        return Model
    if name == "Logger":
        from .logger import Logger

        return Logger
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
