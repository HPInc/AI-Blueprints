import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class AcceleratedInferenceRunner:
    """
    A lightweight inference runner that loads a base model using a ModelSelector
    and optionally applies LoRA fine-tuned weights.

    Designed for accelerated local inference with support for mixed precision (e.g., float16).
    
    Attributes:
        model_selector (ModelSelector): Provides model loading utilities.
        finetuned_path (str, optional): Path to the fine-tuned LoRA adapter weights.
        device (str): Device to run inference on ("cuda" or "cpu").
        dtype (torch.dtype): Data type for inference (default: torch.float16).
        model (PreTrainedModel): Loaded transformer model instance.
        tokenizer (PreTrainedTokenizer): Loaded tokenizer instance.
    """

    def __init__(self, model_selector, finetuned_path=None, dtype=torch.float16):
        """
        Initializes the AcceleratedInferenceRunner.

        Args:
            model_selector (ModelSelector): Instance of a ModelSelector used to load the model.
            finetuned_path (str, optional): Path to a directory containing LoRA fine-tuned weights.
            dtype (torch.dtype, optional): Torch data type for inference precision. Defaults to torch.float16.
        """
        self.model_selector = model_selector
        self.finetuned_path = finetuned_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        self.model = None
        self.tokenizer = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("AcceleratedInferenceRunner")

    def load_model(self):
        """
        Loads the base model and tokenizer from the ModelSelector,
        and applies LoRA fine-tuned weights if provided.

        Raises:
            FileNotFoundError: If LoRA adapter config is expected but missing.
        """
        self.logger.info("🔄 Loading tokenizer and base model from ModelSelector...")

        model_path = self.model_selector.format_model_path(self.model_selector.model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype
        ).to(self.device)

        if self.finetuned_path:
            adapter_config_path = os.path.join(self.finetuned_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                self.logger.info("🎯 Applying LoRA fine-tuned weights...")
                self.model = PeftModel.from_pretrained(self.model, self.finetuned_path)
            else:
                self.logger.warning(f"⚠️ No adapter_config.json found at {self.finetuned_path}. Skipping LoRA application.")

        self.model = self.model.eval()
        self.logger.info("✅ Model loaded and ready for inference.")

    def infer(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        """
        Runs inference for a given prompt using the loaded model.

        Args:
            prompt (str): Input prompt text.
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 100.
            temperature (float, optional): Sampling temperature (controls randomness). Defaults to 0.7.

        Returns:
            str: The generated output text from the model.
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()

        self.logger.info(f"🔍 Running inference for prompt (truncated): {prompt[:80]}...")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.logger.info("✅ Inference complete.")
        return result
