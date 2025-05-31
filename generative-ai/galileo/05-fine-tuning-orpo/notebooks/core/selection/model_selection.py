import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError


class ModelAccessException(Exception):
    """
    Custom exception raised when access to a Hugging Face model repository is restricted.
    """

    def __init__(self, model_id, message="Access to this model is restricted."):
        self.model_id = model_id
        self.message = f"{message} Please request access at: https://huggingface.co/{model_id}"
        super().__init__(self.message)


class ModelSelector:
    """
    Handles the selection, download, loading, and compatibility checking of pre-trained LLMs
    from Hugging Face. Supports offline storage, structured logging, and ORPO compatibility validation.

    Attributes:
        model_list (list[str]): List of allowed Hugging Face model IDs.
        base_local_dir (str): Local directory where models are stored.
        model_id (str): Currently selected model ID.
        model (PreTrainedModel): Loaded Hugging Face model instance.
        tokenizer (PreTrainedTokenizer): Loaded tokenizer instance.
    """

    def __init__(self, model_list=None, base_local_dir=None):
        """
        Initializes the ModelSelector.

        Args:
            model_list (list[str], optional): List of supported model IDs. Defaults to a curated list.
            base_local_dir (str, optional): Directory to store downloaded models. Defaults to '../../../local/models'.
        """
        self.model_list = model_list or [
            "mistralai/Mistral-7B-Instruct-v0.1",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "google/gemma-7b-it",
            "google/gemma-3-1b-it"
        ]
        self.base_local_dir = base_local_dir or os.path.join("..", "..", "..", "local", "models")
        self.model_id = None
        self.model = None
        self.tokenizer = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ModelSelector")

    def log(self, message: str):
        """
        Logs a message using the ModelSelector logger.

        Args:
            message (str): Message to log.
        """
        self.logger.info(f"[ModelSelector] {message}")

    def format_model_path(self, model_id: str) -> str:
        """
        Converts a Hugging Face model ID into a valid local directory path.

        Args:
            model_id (str): Hugging Face model repo ID.

        Returns:
            str: Local filesystem path.
        """
        model_dir_name = model_id.replace("/", "__")
        return os.path.join(self.base_local_dir, model_dir_name)

    def select_model(self, model_id: str):
        """
        Main entry point to select, download, load, and validate a model.

        Args:
            model_id (str): Hugging Face model ID.

        Raises:
            ValueError: If the model is not allowed.
        """
        self.log(f"Selected model: {model_id}")
        if model_id not in self.model_list:
            raise ValueError(f"{model_id} is not a valid option in the model list.")

        self.model_id = model_id
        local_path = self.download_model()
        self.load_model(local_path)
        self.check_compatibility()

    def download_model(self) -> str:
        """
        Downloads a model snapshot from Hugging Face Hub into the local directory.

        Returns:
            str: Path to the local model directory.

        Raises:
            ModelAccessException: If access to the model is restricted.
            RuntimeError: For unexpected download failures.
        """
        model_path = self.format_model_path(self.model_id)
        self.log(f"Downloading model snapshot to: {model_path}")

        try:
            snapshot_download(
                repo_id=self.model_id,
                local_dir=model_path,
                resume_download=True,
                etag_timeout=60,
                local_dir_use_symlinks=False
            )
            self.log(f"✅ Model downloaded successfully to: {model_path}")
            return model_path

        except HfHubHTTPError as e:
            if "401" in str(e) or "403" in str(e):
                raise ModelAccessException(self.model_id)
            raise RuntimeError(f"Unexpected Hugging Face HTTP error: {e}")
        except Exception as e:
            raise RuntimeError(f"Download failed for {self.model_id}: {e}")

    def load_model(self, model_path: str):
        """
        Loads the model and tokenizer from a given local path.

        Args:
            model_path (str): Path to the local model directory.

        Raises:
            RuntimeError: If loading fails.
        """
        self.log(f"Loading model and tokenizer from: {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model/tokenizer from {model_path}: {e}")

    def check_compatibility(self):
        """
        Validates if the loaded model supports ORPO (chat template required).

        Raises:
            ValueError: If the model lacks a chat_template.
        """
        self.log("Checking model for ORPO compatibility...")
        if getattr(self.tokenizer, "chat_template", None) is None:
            raise ValueError(f"The model '{self.model_id}' is missing a chat_template.")
        self.log(f"✅ Model '{self.model_id}' is ORPO-compatible.")

    def get_model(self):
        """
        Returns the loaded model instance.

        Returns:
            PreTrainedModel: Loaded model.
        """
        return self.model

    def get_tokenizer(self):
        """
        Returns the loaded tokenizer instance.

        Returns:
            PreTrainedTokenizer: Loaded tokenizer.
        """
        return self.tokenizer
