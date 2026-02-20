"""
MLflow Loader Layer — _load_pyfunc entry point.

What is this file for?
    When MLflow loads a saved model (e.g., during `mlflow models serve`),
    it calls the function `_load_pyfunc(data_path)` from whichever module
    was registered as `loader_module` in the logger.

    This function is responsible for:
    1. Reading artifacts that were saved alongside the model (config, docs, etc.)
    2. Constructing and returning a fully initialized Model instance

    MLflow then wraps the returned object with its PythonModel interface,
    so the `predict()` method becomes callable via HTTP.

    Reference: https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModel

v2.0.0 Architecture Reminder:
    loader.py (you are here)    ← "Loader Layer"  — tells MLflow how to reconstruct the model
    models/*.py                 ← "Logic Layer"   — per-capability Model classes
    logger.py                   ← "Registry Layer" — knows how to save everything to MLflow

    loader.py reads config["capability"] to select the right Model class.
    It has zero inference code — just routing and instantiation.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_pyfunc(data_path: str):
    """
    Entry point called by MLflow when loading a saved model.

    This function is NOT called during normal notebook usage.
    It is only invoked when running:
        mlflow models serve -m models:/educational-quickstart/1 -p 5002
    or when loading via `mlflow.pyfunc.load_model(...)`.

    How artifacts are organized inside data_path:
        data_path/
        ├── config.yaml       ← blueprint configuration
        ├── secrets.yaml      ← optional: API keys, tokens (not committed to git)
        ├── docs/             ← optional: sample documents for document analyzer
        ├── demo/             ← Streamlit assets (CSS, logos, etc.)
        └── models/           ← optional: small specialized models

    Args:
        data_path: Filesystem path to the directory where model artifacts were saved.
                   This is set automatically by MLflow — you do not pass it manually.

    Returns:
        An initialized Model instance ready to handle predict() calls.
    """
    from src.utils import load_config

    data_path = Path(data_path)

    # ── 1. Load configuration ────────────────────────────────────────────────
    config_path = data_path / "config.yaml"
    if not config_path.exists():
        logger.warning(f"⚠️ config.yaml not found at {config_path}. Using empty config.")
        config = {}
    else:
        config = load_config(str(config_path))
        logger.info(f"✅ Loaded config from {config_path}")

    # ── 2. Load optional secrets ─────────────────────────────────────────────
    secrets_path = data_path / "secrets.yaml"
    secrets: dict = {}
    if secrets_path.exists():
        secrets = load_config(str(secrets_path))
        logger.info("✅ Loaded secrets.yaml (contents not logged for security)")
    else:
        logger.info("ℹ️ No secrets.yaml found — running without external API integrations")

    # ── 3. Resolve docs path ──────────────────────────────────────────────────
    docs_path_candidate = data_path / "docs"
    docs_path = str(docs_path_candidate) if docs_path_candidate.exists() else None

    # ── 4. Resolve model path ─────────────────────────────────────────────────
    # MODEL_ARTIFACTS_PATH is an environment variable set by the MLflow serving container.
    # When running locally on Jupyter, you typically set model_path via config.yaml.
    model_path = os.environ.get(
        "MODEL_ARTIFACTS_PATH",
        config.get("model_path", "")
    )

    if not model_path:
        logger.warning(
            "⚠️ model_path is not set.\n"
            "   Set it in config.yaml OR as the MODEL_ARTIFACTS_PATH environment variable.\n"
            "   LLM will be unavailable until this is configured."
        )

    # ── 5. Select the right Model class based on config["capability"] ────────
    # Each config YAML has a "capability" key that tells loader.py which Model
    # class to instantiate. This is what enables per-capability model registration.
    capability = config.get("capability", "chatbot")

    if capability == "image_gen":
        from src.mlflow.models.image_gen import ImageGenModel as Model
    elif capability == "document":
        from src.mlflow.models.document import DocumentModel as Model
    elif capability == "voice":
        from src.mlflow.models.voice import VoiceModel as Model
    else:
        # Default: chatbot (covers "chatbot" and any unrecognized capability string)
        from src.mlflow.models.chatbot import ChatbotModel as Model

    logger.info(f"Building {Model.__name__}: model_path={model_path!r}, docs_path={docs_path!r}")

    # ── 6. Instantiate and return the Model ──────────────────────────────────
    # All Model classes share the same constructor signature for loader compatibility.
    return Model(
        config=config,
        docs_path=docs_path,
        model_path=model_path,
        secrets=secrets,
    )
