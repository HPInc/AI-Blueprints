import os
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    ClapModel,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from src.utils import get_models_dir, format_model_path, setup_model_environment, logger


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
            "mispeech/midashenglm-7b",
            "Qwen/Qwen2.5-Omni-7B",
            "laion/clap-htsat-unfused",
            "MoonshotAI/Kimi-Audio",
        ]

        # Set up model environment (HF cache, etc.)
        setup_model_environment()

        self.datafabric_base = "/home/jovyan/datafabric"
        self.base_local_dir = str(get_models_dir())
        self.model_id: str | None = None
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.logger = logger

    def log(self, message: str):
        self.logger.info(f"[ModelSelector] {message}")

    def format_model_path(self, model_id: str) -> str:
        """Converts a repo ID into a local directory name using centralized utility."""
        return str(format_model_path(model_id))

    def _check_datafabric_path(self, model_id: str) -> str | None:
        """Check if model exists in datafabric using standard AI Studio paths."""
        # Convert model ID to datafabric path format
        if model_id == "Qwen/Qwen2.5-Omni-7B":
            datafabric_path = os.path.join(self.datafabric_base, "Qwen2.5-Omni-7B")
        elif model_id == "laion/clap-htsat-unfused":
            datafabric_path = os.path.join(self.datafabric_base, "clap-htsat-unfused")
        else:
            # General case: replace '/' with '-' for other models
            safe_name = model_id.replace("/", "-")
            datafabric_path = os.path.join(self.datafabric_base, safe_name)

        return datafabric_path if Path(datafabric_path).exists() else None

    def select_model(self, model_id: str):
        """Downloads and validates the selected model."""
        self.log(f"Selected model: {model_id}")
        if model_id not in self.model_list:
            raise ValueError(f"{model_id} is not a valid option in the model list.")

        self.model_id = model_id

        # First, try to load from datafabric
        datafabric_path = self._check_datafabric_path(model_id)
        if datafabric_path and Path(datafabric_path).exists():
            self.log(f"📁 Loading from datafabric: {datafabric_path}")
            local_path = datafabric_path
        else:
            self.log(f"📡 Datafabric path not found, downloading: {model_id}")
            local_path = self.download_model()

        if self.model_id == "Qwen/Qwen2.5-Omni-7B":
            self.load_qwen_model(local_path)
        elif self.model_id == "laion/clap-htsat-unfused":
            self.load_clap_model(local_path)
        else:
            self.load_model(local_path)

    def download_model(self) -> str:
        """
        Downloads the snapshot and returns the local path.
        Falls back to creating the full directory tree if FileNotFoundError occurs.
        """
        model_path = format_model_path(self.model_id)
        self.log(f"⬇️ Downloading model {self.model_id} to → {str(model_path)}")

        try:
            # Ensure the directory exists
            model_path.mkdir(parents=True, exist_ok=True)

            snapshot_download(
                repo_id=self.model_id,
                local_dir=str(model_path),
                resume_download=True,
                etag_timeout=60,
                local_dir_use_symlinks=False,
                token=os.environ.get("AIS_HUGGINGFACE_API_KEY"),
            )

        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                raise ModelAccessException(
                    self.model_id,
                    "You need to be authenticated and have access permission.",
                )
            elif e.response.status_code == 403:
                raise ModelAccessException(self.model_id)
            else:
                raise RuntimeError(f"Unexpected Hugging Face HTTP error: {e}")

        except Exception as e:
            raise RuntimeError(f"Download failed for {self.model_id}: {e}")

        self.log(f"✅ Model downloaded successfully to: {model_path}")
        return str(model_path)

    def load_qwen_model(self, model_path: str):
        """Loads the Qwen model and processor from disk."""
        self.log(f"Loading Qwen model and processor from: {model_path}")
        try:
            self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
            self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                model_path, torch_dtype="auto", device_map="auto"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Qwen model/processor from {model_path}: {e}"
            )

    def load_clap_model(self, model_path: str):
        """Loads the CLAP model and processor from disk."""
        self.log(f"Loading CLAP model and processor from: {model_path}")
        try:
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = ClapModel.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CLAP model/processor from {model_path}: {e}"
            )

    def load_model(self, model_path: str):
        """Loads model and tokenizer from disk."""
        self.log(f"Loading model and tokenizer from: {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model/tokenizer from {model_path}: {e}")

    def get_model(self):
        return self.model

    def get_processor(self):
        return self.processor

    def get_tokenizer(self):
        return self.tokenizer
