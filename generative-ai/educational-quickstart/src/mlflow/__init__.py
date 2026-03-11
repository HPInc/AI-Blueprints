"""
MLflow models-from-code implementation for Educational Quickstart.

This module provides the standardized universal structure for MLflow model logging
and loading using the models-from-code approach.

What is "models-from-code"?
    Instead of serializing (pickling) a Python object to disk, MLflow 3.x stores
    a reference to the Python SOURCE CODE of your model loader. When you later
    load the model, MLflow runs that source code to reconstruct the model from
    scratch. This avoids pickle security issues and works better with large models.

Architecture (v2.0.0):
    models/
        chatbot.py   → ChatbotModel   (LLM-based conversational Q&A)
        image_gen.py → ImageGenModel  (SDXL-Turbo text-to-image)
        document.py  → DocumentModel  (chunk-based RAG Q&A)
        voice.py     → VoiceModel     (Whisper + LLM pipeline)
    loader.py        → _load_pyfunc() reads config["capability"] → selects Model class
    logger.py        → Logger.log_model() packages everything into MLflow

Learn more:
    https://mlflow.org/docs/latest/models.html
"""

__all__ = ["ChatbotModel", "ImageGenModel", "DocumentModel", "VoiceModel", "Logger"]


def __getattr__(name):
    """
    Dynamic (lazy) import for Model classes and Logger.

    Why lazy loading?
        Importing heavy dependencies like torch, transformers, and LlamaCpp at
        module load time would slow down every import of this package — even when
        you only need Logger (no GPU needed). Lazy loading defers these imports
        until the class is actually used.

    This is a Python dunder (double-underscore) method that is called automatically
    when you access an attribute that doesn't exist at the module level.
    """
    if name == "ChatbotModel":
        from .models.chatbot import ChatbotModel

        return ChatbotModel
    if name == "ImageGenModel":
        from .models.image_gen import ImageGenModel

        return ImageGenModel
    if name == "DocumentModel":
        from .models.document import DocumentModel

        return DocumentModel
    if name == "VoiceModel":
        from .models.voice import VoiceModel

        return VoiceModel
    if name == "Logger":
        from .logger import Logger

        return Logger
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
