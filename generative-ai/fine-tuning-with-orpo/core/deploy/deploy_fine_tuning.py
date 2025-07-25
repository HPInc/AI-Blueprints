from __future__ import annotations

import logging, re, pandas as pd, torch, mlflow, sys, os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils import get_fine_tuned_models_dir, get_models_dir, get_project_root

from transformers import AutoTokenizer, AutoModelForCausalLM
from mlflow.types import Schema, ColSpec
from mlflow.models import ModelSignature
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

_FULL_WT_RGX = re.compile(r"^(pytorch_model|model)(-\d+)?\.(bin|safetensors)$")


def _dir_has_full_weights(path: Path) -> bool:
    """
    Check if a directory contains a complete model (full weights) rather than LoRA adapters.
    
    Returns True if:
    1. Contains pytorch_model*.bin or model*.safetensors files (standard model weights)
    2. Contains config.json but NOT adapter_config.json (indicates complete model)
    3. Has model.safetensors (common single-file format)
    
    Returns False if:
    1. Contains adapter_config.json (indicates LoRA adapters)
    2. No weight files found
    """
    if not path.is_dir():
        return False
    
    files = [p.name for p in path.iterdir() if p.is_file()]
    
    # If adapter_config.json exists, it's definitely LoRA adapters
    if "adapter_config.json" in files:
        return False
    
    # Check for standard model weight files
    has_model_weights = any(_FULL_WT_RGX.match(fname) for fname in files)
    if has_model_weights:
        return True
    
    # Check for single-file model formats
    if "model.safetensors" in files or "pytorch_model.bin" in files:
        return True
    
    # If config.json exists without adapter_config.json, likely a complete model
    if "config.json" in files and "adapter_config.json" not in files:
        # Additional check: look for any .safetensors or .bin files
        has_weight_files = any(fname.endswith(('.safetensors', '.bin')) for fname in files)
        return has_weight_files
    
    return False


def _is_lora_adapter_dir(path: Path) -> bool:
    """Check if a directory contains LoRA adapters."""
    if not path.is_dir():
        return False
    
    files = [p.name for p in path.iterdir() if p.is_file()]
    return "adapter_config.json" in files


def _as_path(obj) -> Path:
    """Ensure `obj` is Path (do not convert Hub IDs).."""
    return obj if isinstance(obj, Path) else Path(obj)


def _load_tokenizer(src: Union[str, Path]):
    p = _as_path(src)
    if p.exists():
        return AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    hub_id = str(src).replace("__", "/")
    return AutoTokenizer.from_pretrained(hub_id, trust_remote_code=True)


def _load_model(
    src: Union[str, Path],
    device: str = "auto",
    dtype: str = "auto",
    trust_remote: bool = True,
):
    """Load model with adaptive memory optimization."""
    p = _as_path(src)
    
    # Adaptive memory optimization based on available memory and device
    is_cuda = torch.cuda.is_available() and device in ["cuda", "auto"]
    
    # Memory optimization kwargs
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote,
    }
    
    # Adaptive torch_dtype
    if dtype == "auto":
        model_kwargs["torch_dtype"] = torch.float16 if is_cuda else torch.float32
    else:
        model_kwargs["torch_dtype"] = dtype
    
    # Adaptive device mapping for memory efficiency
    if device == "auto":
        if is_cuda:
            model_kwargs["device_map"] = "auto"  # Let transformers handle device placement
        else:
            model_kwargs["device_map"] = None
    elif device == "cuda" and is_cuda:
        model_kwargs["device_map"] = "auto"
    
    if p.exists():
        model = AutoModelForCausalLM.from_pretrained(str(p), **model_kwargs)
    else:
        hub_id = str(src).replace("__", "/")
        logging.info("🌐 Downloading from Hub with adaptive optimization: %s", hub_id)
        model = AutoModelForCausalLM.from_pretrained(hub_id, **model_kwargs)
    
    # Only explicitly move to device if device_map wasn't used
    if "device_map" not in model_kwargs or model_kwargs["device_map"] is None:
        target_device = device if device != "auto" else ("cuda" if is_cuda else "cpu")
        model = model.to(target_device)
    
    return model


class LLMComparisonModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        base_src = context.artifacts["model_no_finetuning"]
        ft_src   = context.artifacts["finetuned_model"]

        # Adaptive device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_src = base_src
        self.ft_src = ft_src
        
        # Model state management
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_type = None
        
        # Pre-cache tokenizers (small memory footprint)
        try:
            self.base_tokenizer = _load_tokenizer(self.base_src)
            
            # Determine fine-tuned model type and tokenizer
            ft_path = _as_path(self.ft_src)
            if ft_path.exists() and ft_path.is_dir():
                if _is_lora_adapter_dir(ft_path):
                    self.ft_is_lora = True
                    self.ft_tokenizer = self.base_tokenizer  # LoRA uses base tokenizer
                    logging.info("🔍 Detected LoRA adapter directory")
                elif _dir_has_full_weights(ft_path):
                    self.ft_is_lora = False
                    self.ft_tokenizer = _load_tokenizer(self.ft_src)
                    logging.info("🔍 Detected complete model directory")
                else:
                    # Fallback: assume it's a complete model if it has config.json
                    config_exists = (ft_path / "config.json").exists()
                    self.ft_is_lora = not config_exists
                    self.ft_tokenizer = _load_tokenizer(self.ft_src) if config_exists else self.base_tokenizer
                    logging.warning(f"🔍 Model type unclear, assuming {'LoRA' if self.ft_is_lora else 'complete'} based on config.json presence")
            else:
                # Assume it's a HuggingFace model ID
                self.ft_is_lora = False
                self.ft_tokenizer = _load_tokenizer(self.ft_src)
                logging.info("🔍 Assuming HuggingFace model ID")
                
            logging.info("🚀 Adaptive model system initialized. Models will be loaded on-demand.")
            logging.info(f"📊 Using device: {self.device}, LoRA mode: {self.ft_is_lora}")
        except Exception as e:
            logging.warning(f"⚠️ Model type detection failed: {e}. Will determine type during loading.")
            self.base_tokenizer = None
            self.ft_tokenizer = None
            self.ft_is_lora = None

    def _clear_model_memory(self):
        """Aggressively clear model from memory with adaptive cleanup."""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.current_model_type = None
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection for memory-constrained environments
            import gc
            gc.collect()
            
            logging.info("🧹 Model memory cleared with adaptive cleanup")

    def _load_model_on_demand(self, use_ft: bool):
        """Load the requested model only when needed, with adaptive memory management."""
        target_type = "ft" if use_ft else "base"
        
        # Return early if correct model is already loaded
        if self.current_model_type == target_type and self.current_model is not None:
            return
        
        # Clear current model from memory first
        self._clear_model_memory()
        
        # Load the requested model with adaptive optimization
        try:
            if use_ft:
                ft_path = _as_path(self.ft_src)
                if ft_path.exists() and ft_path.is_dir():
                    # Determine model type if not already known
                    if self.ft_is_lora is None:
                        is_lora = _is_lora_adapter_dir(ft_path)
                        has_full_weights = _dir_has_full_weights(ft_path)
                        logging.info(f"🔍 Dynamic detection: LoRA={is_lora}, Full weights={has_full_weights}")
                    else:
                        is_lora = self.ft_is_lora
                        has_full_weights = not is_lora
                    
                    if has_full_weights and not is_lora:
                        logging.info("🟢 Loading fine-tuned complete checkpoint")
                        self.current_tokenizer = self.ft_tokenizer or _load_tokenizer(ft_path)
                        self.current_model = _load_model(ft_path, device="auto").eval()
                    elif is_lora:
                        logging.info("🟠 Loading fine-tuned LoRA adapter")
                        # Load base model for LoRA merging
                        base_tokenizer = self.base_tokenizer or _load_tokenizer(self.base_src)
                        base_model = _load_model(self.base_src, device="auto")
                        
                        # Apply LoRA and merge
                        self.current_model = (
                            PeftModel.from_pretrained(
                                base_model, str(ft_path), is_trainable=False
                            )
                            .merge_and_unload()
                            .eval()
                        )
                        self.current_tokenizer = base_tokenizer
                        
                        # Clean up base model immediately
                        del base_model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        # Fallback: try to load as complete model first, then as LoRA
                        try:
                            logging.info("🔄 Attempting to load as complete model (fallback)")
                            self.current_tokenizer = self.ft_tokenizer or _load_tokenizer(ft_path)
                            self.current_model = _load_model(ft_path, device="auto").eval()
                        except Exception as complete_error:
                            logging.warning(f"⚠️ Failed to load as complete model: {complete_error}")
                            logging.info("🔄 Attempting to load as LoRA adapter (fallback)")
                            base_tokenizer = self.base_tokenizer or _load_tokenizer(self.base_src)
                            base_model = _load_model(self.base_src, device="auto")
                            
                            self.current_model = (
                                PeftModel.from_pretrained(
                                    base_model, str(ft_path), is_trainable=False
                                )
                                .merge_and_unload()
                                .eval()
                            )
                            self.current_tokenizer = base_tokenizer
                            
                            del base_model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                else:
                    logging.info("🌐 Loading fine-tuned model from Hub")
                    self.current_tokenizer = self.ft_tokenizer or _load_tokenizer(self.ft_src)
                    self.current_model = _load_model(self.ft_src, device="auto").eval()
            else:
                logging.info("🔵 Loading base model")
                self.current_tokenizer = self.base_tokenizer or _load_tokenizer(self.base_src)
                self.current_model = _load_model(self.base_src, device="auto").eval()
            
            self.current_model_type = target_type
            logging.info(f"✅ {target_type.upper()} model loaded successfully")
            
        except Exception as e:
            logging.error(f"❌ Failed to load {target_type} model: {str(e)}")
            self._clear_model_memory()  # Clear memory on error
            raise

    def predict(self, context, model_input, params=None):
        """Adaptive prediction with memory and performance optimization."""
        try:
            # Handle both DataFrame and dict inputs for backward compatibility
            if hasattr(model_input, 'iloc'):  # DataFrame
                prompt = model_input["prompt"].iloc[0]
                use_ft = model_input["use_finetuning"].iloc[0]
                max_tok = model_input.get("max_tokens", pd.Series([128])).iloc[0]
            else:  # Dict or other format
                prompt = model_input.get("prompt", "")
                use_ft = model_input.get("use_finetuning", False)
                max_tok = model_input.get("max_tokens", 128)

            # Load the appropriate model on-demand
            self._load_model_on_demand(use_ft)

            # Prepare inputs with device handling
            inputs = self.current_tokenizer(prompt, return_tensors="pt")
            
            # Move inputs to model device
            model_device = next(self.current_model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            # Generate with adaptive optimization
            with torch.no_grad():
                # Use autocast if available and on CUDA for memory efficiency
                if model_device.type == "cuda" and torch.cuda.is_available():
                    try:
                        with torch.cuda.amp.autocast():
                            ids = self.current_model.generate(
                                **inputs,
                                max_new_tokens=max_tok,
                                do_sample=False,
                                pad_token_id=self.current_tokenizer.eos_token_id,
                                use_cache=True,
                            )
                    except:
                        # Fallback without autocast if it fails
                        ids = self.current_model.generate(
                            **inputs,
                            max_new_tokens=max_tok,
                            do_sample=False,
                            pad_token_id=self.current_tokenizer.eos_token_id,
                            use_cache=True,
                        )
                else:
                    ids = self.current_model.generate(
                        **inputs,
                        max_new_tokens=max_tok,
                        do_sample=False,
                        pad_token_id=self.current_tokenizer.eos_token_id,
                        use_cache=True,
                    )

            # Decode response
            txt = self.current_tokenizer.decode(ids[0], skip_special_tokens=True)
            return pd.DataFrame({"response": [txt]})
            
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            self._clear_model_memory()  # Clear memory on error
            raise


def register_llm_comparison_model(
    model_base_path: str,
    model_finetuned_path: str,
    experiment: str,
    run_name: str,
    registry_name: str,
    config_path: str = "../configs/config.yaml",
):
    """
    Register an adaptive LLM comparison model with MLflow.
    
    This model automatically adapts to memory constraints and available hardware,
    providing robust deployment across different environments.
    
    Args:
        model_base_path: Path to base model (can be relative to project)
        model_finetuned_path: Path to fine-tuned model (can be relative to project)
        experiment: MLflow experiment name
        run_name: MLflow run name
        registry_name: Model registry name
        config_path: Path to configuration file (default: ../configs/config.yaml)
    """
    # Validate and resolve paths
    def resolve_model_path(path_str: str) -> str:
        """Resolve model path, making it project-relative if needed."""
        path = Path(path_str)
        
        # If absolute path and exists, use as-is
        if path.is_absolute() and path.exists():
            return str(path)
            
        # If relative path, try to resolve relative to project directories
        project_root = get_project_root()
        
        # Try models directory
        models_path = get_models_dir() / path_str
        if models_path.exists():
            return str(models_path)
            
        # Try fine-tuned models directory
        ft_path = get_fine_tuned_models_dir() / path_str
        if ft_path.exists():
            return str(ft_path)
            
        # Try relative to project root
        root_path = project_root / path_str
        if root_path.exists():
            return str(root_path)
            
        # If it's a HuggingFace model ID, return as-is
        if "/" in path_str and not path_str.startswith("../"):
            return path_str
            
        # Return original path and let downstream handle the error
        logging.warning(f"Could not resolve model path: {path_str}")
        return path_str
    
    resolved_base_path = resolve_model_path(model_base_path)
    resolved_ft_path = resolve_model_path(model_finetuned_path)
    
    logging.info(f"Resolved base model path: {resolved_base_path}")
    logging.info(f"Resolved fine-tuned model path: {resolved_ft_path}")
    
    core = Path(__file__).resolve().parent.parent
    src = core.parent / "src"
    (core / "__init__.py").touch(exist_ok=True)
    (src / "__init__.py").touch(exist_ok=True)

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        signature = ModelSignature(
            inputs=Schema(
                [
                    ColSpec("string",  "prompt"),
                    ColSpec("boolean", "use_finetuning"),
                    ColSpec("integer", "max_tokens"),
                ]
            ),
            outputs=Schema([ColSpec("string", "response")]),
        )

        mlflow.pyfunc.log_model(
            artifact_path="llm_serving_model",
            python_model=LLMComparisonModel(),
            artifacts={
                "model_no_finetuning": resolved_base_path,
                "finetuned_model":     resolved_ft_path,
                "config": str(Path(config_path).resolve()),
            },
            signature=signature,
            code_paths=[str(core), str(src)],
            pip_requirements=[
                "torch",
                "transformers==4.51.3",
                "peft==0.15.2",
                "accelerate==1.6.0",
                "mlflow",
                "pandas",
            ],
        )

        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/llm_serving_model",
            name=registry_name,
        )
        logging.info("✅ Adaptive LLM comparison model registered as `%s` (run %s)", registry_name, run.info.run_id)
