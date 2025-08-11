"""
MLflow loader for TextGenerationModel.

Loader Module
- Handles loading of model artifacts and configuration
- Initializes domain components (LLMs, config, etc.)
- Constructs and returns the Model instance (not service!)
- Manages artifact path resolution
- NO business logic - pure integration layer
"""

import logging
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Any

import mlflow
import yaml
from mlflow.models import infer_signature

from .text_generation_model import TextGenerationModel


def _add_project_to_syspath():
    """
    Ensure core and src are on sys.path for imports when model is loaded 
    inside MLflow scoring server.
    """
    core_path = Path(__file__).resolve().parent.parent
    (core_path / "__init__.py").touch(exist_ok=True)
    sys.path.insert(0, str(core_path))

    src_path = next(
        (p / "src" for p in [core_path, *core_path.parents] if (p / "src").is_dir()),
        None,
    )
    if src_path:
        sys.path.insert(0, str(src_path))

    sys.path.insert(0, str(core_path.parent))
    return core_path, src_path


def _load_llm(artifacts: Dict[str, str]):
    """Load the LlamaCpp model from artifacts."""
    from src.utils import (
        configure_hf_cache,
        configure_proxy,
        load_config
    )
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain_community.llms import LlamaCpp

    if hasattr(LlamaCpp, "model_rebuild"):
        LlamaCpp.model_rebuild()

    cfg_dir = Path(artifacts["config"]).parent
    cfg = load_config(cfg_dir / "config.yaml")

    model_path = artifacts.get("llm") or ""
    if not model_path:
        raise RuntimeError("Missing *.gguf artifact for the LLM.")

    configure_hf_cache()
    configure_proxy(cfg)

    start = time.perf_counter()
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=int(cfg.get("n_gpu_layers", 1)),  # 0 → CPU-only
        n_batch=256,
        n_ctx=4096,
        max_tokens=1024,
        f16_kv=True,
        temperature=0.2,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
        streaming=False,
    )
    logging.info("🔹 LlamaCpp loaded in %.1fs", time.perf_counter() - start)
    return llm


def _load_model(model_uri):
    """
    Load model components and return configured Model instance.
    
    Args:
        model_uri: URI to the MLflow model
        
    Returns:
        TextGenerationModel: Configured model instance
    """
    import logging
    
    # Set up project paths
    _add_project_to_syspath()
    
    # Load artifacts from MLflow
    artifacts = {}
    if hasattr(mlflow.artifacts, 'download_artifacts'):
        artifact_path = mlflow.artifacts.download_artifacts(model_uri)
        artifacts_dir = Path(artifact_path)
        
        # Map expected artifacts
        artifacts = {
            "config": str(artifacts_dir / "config"),
            "llm": str(artifacts_dir / "llm"),
        }
        
        # Handle optional artifacts
        if (artifacts_dir / "secrets").exists():
            artifacts["secrets"] = str(artifacts_dir / "secrets")
        if (artifacts_dir / "demo").exists():
            artifacts["demo"] = str(artifacts_dir / "demo")
    
    # Load configuration
    config_path = artifacts["config"]
    if os.path.exists(config_path):
        if os.path.isfile(config_path):
            # Direct config file
            with open(config_path) as file:
                config = yaml.safe_load(file)
        else:
            # Config directory - look for config.yaml
            config_file = Path(config_path) / "config.yaml"
            if config_file.exists():
                with open(config_file) as file:
                    config = yaml.safe_load(file)
            else:
                config = {}
    else:
        config = {}
        logging.warning(f"Configuration not found at {config_path}")

    # Load LLM
    llm = _load_llm(artifacts)
    
    # Create and return Model instance
    model = TextGenerationModel(llm=llm, config=config)
    
    return model


def save_model(
    model_instance,
    model_path: str,
    llm_artifact: str,
    config_path: str,
    secrets_dict: Dict = None,
    demo_folder: str = None,
    sample_input=None, 
    sample_output=None
):
    """
    Save model using models-from-code approach.
    
    Args:
        model_instance: TextGenerationModel instance
        model_path: Path to save the model
        llm_artifact: Path to the LLM artifact
        config_path: Path to the configuration file
        secrets_dict: Dict with secrets to persist as YAML (optional)
        demo_folder: Path to the demo folder (optional)
        sample_input: Sample input for signature inference
        sample_output: Sample output for signature inference
    """
    # Prepare artifacts
    artifacts = {
        "config": str(Path(config_path).resolve()),
        "llm": llm_artifact,
    }

    if secrets_dict:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.safe_dump(secrets_dict, tmp)
        tmp.close()
        artifacts["secrets"] = tmp.name
        logging.info(f"Secrets artifact written to temporary file {tmp.name}")
    
    # Add demo folder to artifacts if provided
    if demo_folder:
        artifacts["demo"] = str(Path(demo_folder).resolve())

    # Infer signature if sample data provided
    signature = infer_signature(sample_input, sample_output) if sample_input is not None else None

    # Get code paths for models-from-code
    core_path, src_path = _add_project_to_syspath()
    code_paths = [str(core_path)] + ([str(src_path)] if src_path else [])

    # Save using models-from-code
    mlflow.models.save_model(
        path=model_path,
        loader_module="text_generation_loader",  # This module
        data_path=None,  # No additional data needed
        signature=signature,
        conda_env=None,  # Will use pip_requirements
        pip_requirements="../requirements.txt",
        code_paths=code_paths
    )
