"""
Utility functions for AI Studio GenAI Templates.

This module contains common functions used across notebooks in the project,
including configuration loading, model initialization, and Galileo integration.
"""

import os
import yaml
import importlib.util
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional, List
from .trt_llm_langchain import TensorRTLangchain
from langchain.schema.document import Document
from langchain.chat_models import ChatOpenAI
import mlflow
import math
import matplotlib.pyplot as plt
from PIL import Image as PILImage

logger = logging.getLogger("multimodal_rag_logger")

def configure_hf_cache(cache_dir: str = "/home/jovyan/local/hugging_face") -> None:
    """
    Configure HuggingFace cache directories to persist models locally.

    Args:
        cache_dir: Base directory for HuggingFace cache. Defaults to "/home/jovyan/local/hugging_face".
    """
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")

def log_asset_status(asset_path: str, asset_name: str, success_message: str, failure_message: str) -> None:
    """
    Logs the status of a given asset based on its existence.

    Parameters:
        asset_path (str): File or directory path to check.
        asset_name (str): Name of the asset for logging context.
        success_message (str): Message to log if asset exists.
        failure_message (str): Message to log if asset does not exist.
    """
    if Path(asset_path).exists():
        logger.info(f"{asset_name} is properly configured. {success_message}")
    else:
        logger.info(f"{asset_name} is not properly configured. {failure_message}")

def multimodal_rag_asset_status(
    local_model_path: str,
    config_path: str,
    secrets_path: str,
    wiki_metadata_dir: str,
    context_dir: str,
    chroma_dir: str,
    cache_dir: str,
    manifest_path: str
) -> None:
    """Logs the configuration status of all assets required for the multimodal RAG notebook."""

    log_asset_status(
        asset_path=local_model_path,
        asset_name="Local Model",
        success_message="",
        failure_message="Please check if the local model was properly configured in your project in your datafabrics folder."
    )
    log_asset_status(
        asset_path=config_path,
        asset_name="Config",
        success_message="",
        failure_message="Please check if the configs.yaml was properly configured in your project on AI Studio."
    )
    log_asset_status(
        asset_path=secrets_path,
        asset_name="Secrets",
        success_message="",
        failure_message="Please check if the secrets.yaml was properly configured in your project on AI Studio. If you are using secrets manager you can ignore this message."
    )
    log_asset_status(
        asset_path=wiki_metadata_dir,
        asset_name="wiki_flat_structure.json",
        success_message="",
        failure_message="Place JSON Wiki Pages in data/"
    )
    log_asset_status(
        asset_path=context_dir,
        asset_name="CONTEXT",
        success_message="",
        failure_message="Please check if CONTEXT path was downloaded correctly in your project on AI Studio."
    )
    log_asset_status(
        asset_path=chroma_dir,
        asset_name="CHROMA",
        success_message="",
        failure_message="Please check if CHROMA path was downloaded correctly in your project on AI Studio."
    )
    log_asset_status(
        asset_path=cache_dir,
        asset_name="CACHE",
        success_message="",
        failure_message="Please check if the CHROMA/CACHE path was properly configured in your project on AI Studio."
    )
    log_asset_status(
        asset_path=manifest_path,
        asset_name="MANIFEST",
        success_message="",
        failure_message="Please check if the MANIFEST path was properly configured in your project on AI Studio."
    )
    
def _load_yaml_file(path: str, name: str) -> Dict[str, Any]:
    """
    Helper to load any YAML file with a consistent error message.
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"{name} file not found at: {abs_path}")
    with open(abs_path, 'r') as f:
        return yaml.safe_load(f) or {}

def load_config(config_path: str = "../../configs/config.yaml") -> Dict[str, Any]:
    """
    Load application configuration from a YAML file.
    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        Configuration dict.

    Raises:
        FileNotFoundError: If the config file is not found.
    """
    return _load_yaml_file(config_path, "config.yaml")

def load_secrets(secrets_path: str = "../../configs/secrets.yaml") -> Dict[str, Any]:
    """
    Load application secrets from a YAML file.
    Args:
        secrets_path: Path to the secrets YAML file.

    Returns:
        Secrets dict (may be empty if you’re using a secrets manager setup).

    Raises:
        FileNotFoundError: If the secrets file is not found.
    """
    # If using an external secrets manager, you can create
    # an (empty) secrets.yaml stub in this path to satisfy the loader.
    return _load_yaml_file(secrets_path, "secrets.yaml")

def load_mm_docs_clean(json_path: Path, img_dir: Path) -> List[Document]:
    """
    Load wiki Markdown + image references from *json_path*.
    • Filters out images with bad extensions or missing files.
    • Logs the first 20 broken refs.
    • Returns a list[Document] where metadata = {source, images}
    """
    VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

    bad_imgs, docs = [], []

    rows = json.loads(json_path.read_text("utf-8"))
    for row in rows:
        images_ok = []
        for name in row.get("images", []):
            if not name: # empty
                bad_imgs.append((row["path"], name, "empty"))
                continue
            ext = Path(name).suffix.lower()
            if ext not in VALID_EXTS: # unsupported ext
                bad_imgs.append((row["path"], name, f"ext {ext}"))
                continue
            img_path = img_dir / name
            if not img_path.is_file(): # missing on disk
                bad_imgs.append((row["path"], name, "missing file"))
                continue
            images_ok.append(name)

        docs.append(
            Document(
                page_content=row["content"],
                metadata={"source": row["path"], "images": images_ok},
            )
        )

    # ---- summary logging ----------------------------------------------------
    if bad_imgs:
        logger.warning("⚠️ %d broken image refs filtered out", len(bad_imgs))
        for src, name, reason in bad_imgs[:20]:
            logger.debug("  » %s → %s (%s)", src, name or "<EMPTY>", reason)
    else:
        logger.info("✅ no invalid image refs found")

    return docs

def display_images(image_paths: List[str], max_cols: int = 4):
    """
    Opens and displays a list of images from their file paths in a grid.

    Args:
        image_paths: A list of file paths to the images.
        max_cols: The maximum number of columns in the display grid.
    """
    if not image_paths:
        print("▶ No images to display.")
        return

    print(f"🖼️ Displaying {len(image_paths)} image(s):")
    
    # --- Calculate grid size ---
    num_images = len(image_paths)
    cols = min(num_images, max_cols)
    rows = math.ceil(num_images / cols)

    # --- Create figure and axes ---
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    # Flatten the axes array for easy iteration, and handle the case of a single image
    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # --- Loop through images and display them ---
    for i, path in enumerate(image_paths):
        ax = axes[i]
        try:
            # Open the image file
            img = PILImage.open(path)
            
            # Display the image on the appropriate subplot
            ax.imshow(img)
            ax.set_title(path.split('/')[-1], fontsize=8) # Use filename as title
            
        except FileNotFoundError:
            ax.text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, 'Error loading image', ha='center', va='center', fontsize=9)
            print(f"  - Error loading image '{path}': {e}")
        
        ax.axis('off') # Hide the x/y axis for a cleaner look

    # --- Hide any unused subplots ---
    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def mlflow_evaluate_setup(
    secrets: dict,
    mlflow_tracking_uri: str = "/phoenix/mlflow"
) -> None:
    """
    Prepare the environment for MLflow LLM-judge evaluation in MLflow 2.21.2.
    Args:
        secrets (dict): Dictionary loaded from your secrets.yaml.
        mlflow_tracking_uri (str, optional): If provided, sets MLflow's tracking URI.
    """
    # Set MLflow tracking URI
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    else:
        raise ValueError("❌ Tracking URI is missing")

    # Informational log
    print(f"✅ Environment ready for MLflow evaluation.")