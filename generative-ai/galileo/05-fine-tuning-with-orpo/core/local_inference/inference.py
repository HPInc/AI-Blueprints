import os
import torch
import yaml
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

class InferenceRunner:
    """
    A utility class for loading a language model from a local snapshot path
    and performing inference with GPU/CPU-aware configuration.

    This class automatically detects available GPU resources and loads an
    appropriate Accelerate configuration, supporting seamless model switching
    for inference workflows.

    Attributes:
        model_selector (ModelSelector): Selector for resolving the model snapshot path.
        model (PreTrainedModel): Loaded Hugging Face transformer model.
        tokenizer (PreTrainedTokenizer): Loaded tokenizer corresponding to the model.
        device (str): Device used for inference ("cuda" if available, otherwise "cpu").
        config_dir (str): Directory containing Accelerate YAML configuration files.
        config (dict): Loaded Accelerate configuration.
    """

    def __init__(self, model_selector, config_dir="config"):
        """
        Initializes the InferenceRunner.

        Args:
            model_selector (ModelSelector): Instance of ModelSelector containing the model ID.
            config_dir (str, optional): Directory path for Accelerate config YAMLs. Defaults to "config".
        """
        self.model_selector = model_selector
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config_dir = config_dir

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("InferenceRunner")

        # Load optimal configuration on initialization
        self.config = self.load_optimal_config()

    def log(self, message: str):
        """
        Utility method to log messages with a standardized prefix.

        Args:
            message (str): The message to be logged.
        """
        self.logger.info(f"[InferenceRunner] {message}")

    def load_optimal_config(self) -> dict:
        """
        Loads the optimal Accelerate configuration based on the number of available GPUs.

        Returns:
            dict: Parsed YAML content of the selected configuration.
        """
        num_gpus = torch.cuda.device_count()

        if num_gpus >= 2:
            config_file = os.path.join(self.config_dir, "default_config_multi-gpu.yaml")
            self.log(f"Detected {num_gpus} GPUs, loading multi-GPU configuration.")
        elif num_gpus == 1:
            config_file = os.path.join(self.config_dir, "default_config_one-gpu.yaml")
            self.log("Detected 1 GPU, loading single-GPU configuration.")
        else:
            config_file = os.path.join(self.config_dir, "cpu_config.yaml")
            self.log("No GPU detected, loading CPU configuration.")

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def load_model_from_snapshot(self):
        """
        Loads the model and tokenizer from the resolved snapshot path.

        Raises:
            RuntimeError: If loading the model or tokenizer fails.
        """
        model_path = self.model_selector.format_model_path(self.model_selector.model_id)
        self.log(f"Loading model and tokenizer from snapshot at: {model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model/tokenizer: {e}")

    def infer(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        """
        Runs a single inference step for a given prompt using the loaded model.

        Args:
            prompt (str): The input prompt text.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
            temperature (float, optional): Sampling temperature for text generation. Defaults to 0.7.

        Returns:
            str: The generated output text.
        """
        if self.model is None or self.tokenizer is None:
            self.load_model_from_snapshot()

        self.log(f"Running inference on input: {prompt[:80]}...")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.log("Inference completed.")
        return decoded
