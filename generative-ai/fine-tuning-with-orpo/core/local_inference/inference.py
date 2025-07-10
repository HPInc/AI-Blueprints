import os
import sys
import torch
import yaml
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))
from utils import get_config_dir

class InferenceRunner:
  
    def __init__(self, model_selector, config_dir=None):
        self.model_selector = model_selector
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if config_dir is None:
            # Use centralized path utility
            self.config_dir = get_config_dir()
        else:
            self.config_dir = Path(config_dir).expanduser().resolve()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("InferenceRunner")

        self.config = self.load_optimal_config()

    def log(self, message: str):
        self.logger.info(f"[InferenceRunner] {message}")

    def load_optimal_config(self) -> dict:
        
        num_gpus = torch.cuda.device_count()

        if num_gpus >= 2:
            cfg_name = "default_config_multi-gpu.yaml"
            self.log(f"Detected {num_gpus} GPUs, loading multi-GPU config.")
        elif num_gpus == 1:
            cfg_name = "default_config_one-gpu.yaml"
            self.log("Detected 1 GPU, loading single-GPU config.")
        else:
            cfg_name = "cpu_config.yaml"
            self.log("No GPU detected, loading CPU config.")

        config_file = self.config_dir / cfg_name

        if not config_file.exists():
            raise FileNotFoundError(f"Not found: {config_file}")

        with config_file.open("r") as f:
            return yaml.safe_load(f)

    def load_model_from_snapshot(self):
        model_path = self.model_selector.format_model_path(self.model_selector.model_id)
        self.log(f"Loading model and tokenizer from: {model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Error model/tokenizer: {e}")

    def infer(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        
        if self.model is None or self.tokenizer is None:
            self.load_model_from_snapshot()

        self.log(f"running: {prompt[:80]}...")

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
        self.log("Inferência finalizada.")
        return decoded
