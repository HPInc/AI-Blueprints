"""
Per-capability Model classes for Educational Quickstart.

Architecture (v2.0.0):
    Each Model class encapsulates exactly ONE AI capability, following the same
    constructor signature so that loader.py can instantiate any of them identically.

        ChatbotModel  → LLM-based conversational Q&A
        ImageGenModel → Text-to-image via diffusion pipeline
        DocumentModel → Chunk-based document Q&A (retrieval-augmented generation)
        VoiceModel    → Whisper speech-to-text + LLM response

    The loader selects the right class by reading config["capability"].
    The logger packages the class and its dependencies into an MLflow artifact.

Usage in notebooks (direct instantiation):
    from src.mlflow.models.chatbot import ChatbotModel
    model = ChatbotModel(config=config, model_path="/path/to/model.gguf")
    result = model.predict(pd.DataFrame([{"question": "What is AI?"}]))

Usage at MLflow serve time (automatic, via loader.py):
    loader.py reads config["capability"] → imports the right class → instantiates it.
    You never call this manually.
"""

from .chatbot import ChatbotModel
from .document import DocumentModel
from .image_gen import ImageGenModel
from .voice import VoiceModel

# Registry mapping: capability string → Model class
# loader.py uses this to route requests to the right model at serving time.
MODEL_REGISTRY = {
    "chatbot": ChatbotModel,
    "image_gen": ImageGenModel,
    "document": DocumentModel,
    "voice": VoiceModel,
}

__all__ = [
    "ChatbotModel",
    "ImageGenModel",
    "DocumentModel",
    "VoiceModel",
    "MODEL_REGISTRY",
]
