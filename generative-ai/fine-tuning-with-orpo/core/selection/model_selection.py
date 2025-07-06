import os
import sys
import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from utils import get_models_dir, format_model_path, setup_model_environment


class ModelAccessException(Exception):
    """
    Custom exception raised when access to a Hugging Face model repository is restricted.
    """

    def __init__(self, model_id, message="Access to this model is restricted."):
        self.model_id = model_id
        self.message = (
            f"{message} Please request access at: https://huggingface.co/{model_id}"
        )
        super().__init__(self.message)


class ModelSelector:
    """
    Handles the selection, download, loading, and compatibility checking of
    pre-trained LLMs from Hugging Face. Supports offline storage, structured
    logging, and ORPO compatibility validation.
    """

    def __init__(self, model_list=None, base_local_dir=None):
        """
        Args:
            model_list (list[str], optional): Supported model IDs.
            base_local_dir (str, optional): Base directory for storing models.
                                             Uses project-relative path by default.
        """
        self.model_list = model_list or [
            "mistralai/Mistral-7B-Instruct-v0.1",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "google/gemma-7b-it",
            "google/gemma-3-1b-it",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ]
        
        # Set up model environment (HF cache, etc.)
        setup_model_environment()
        
        self.base_local_dir = str(get_models_dir())
        self.model_id: str | None = None
        self.model = None
        self.tokenizer = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ModelSelector")

    

    def log(self, message: str):
        self.logger.info(f"[ModelSelector] {message}")

    def format_model_path(self, model_id: str) -> str:
        """Converts a repo ID into a local directory name using centralized utility."""
        return str(format_model_path(model_id))

  

    def select_model(self, model_id: str):
        """Downloads, carrega e valida o modelo escolhido."""
        self.log(f"Selected model: {model_id}")
        if model_id not in self.model_list:
            raise ValueError(f"{model_id} is not a valid option in the model list.")

        self.model_id = model_id
        local_path = self.download_model()
        self.load_model(local_path)
        self.check_compatibility()

    def download_model(self) -> str:
        """
        Downloads the snapshot and returns the local path.
        Falls back to creating the full directory tree if FileNotFoundError occurs.
        """
        model_path = format_model_path(self.model_id)
        self.log(f"Downloading model snapshot to: {model_path}")

        try:
            # Ensure the directory exists
            model_path.mkdir(parents=True, exist_ok=True)
            
            snapshot_download(
                repo_id=self.model_id,
                local_dir=str(model_path),
                resume_download=True,
                etag_timeout=60,
                local_dir_use_symlinks=False,
            )

        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                raise ModelAccessException(
                    self.model_id,
                    "You need to be authenticated and have access permission."
                )
            elif e.response.status_code == 403:
                raise ModelAccessException(self.model_id)
            else:
                raise RuntimeError(f"Unexpected Hugging Face HTTP error: {e}")

        except Exception as e:
            raise RuntimeError(f"Download failed for {self.model_id}: {e}")

        self.log(f"✅ Model downloaded successfully to: {model_path}")
        return str(model_path)

    def load_model(self, model_path: str):
        """Loads model and tokenizer from disk."""
        self.log(f"Loading model and tokenizer from: {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model/tokenizer from {model_path}: {e}")

    def check_compatibility(self):
        """Checks if the model supports ORPO/chat-template usage."""
        self.log("Checking model for ORPO compatibility...")
        if getattr(self.tokenizer, "chat_template", None) is None:
            raise ValueError(
                f"The model '{self.model_id}' is missing a chat_template."
            )
        self.log(f"✅ Model '{self.model_id}' is ORPO-compatible.")


    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
