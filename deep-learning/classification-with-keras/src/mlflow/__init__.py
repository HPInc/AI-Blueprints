"""
MLflow models-from-code shim for classification-with-keras blueprint.

This mirrors the vanilla-rag migraton pattern and exposes Model and Logger lazily.
"""

__all__ = ["Model", "Logger"]

def __getattr__(name):
    if name == "Model":
        from .model import Model
        return Model
    if name == "Logger":
        from .logger import Logger
        return Logger
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
