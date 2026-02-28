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

import ctypes
import glob
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────── CUDA library preload ────────────────────────────────────────────────
# Why here, not in voice.py / image_gen.py?
#   loader.py is the *first blueprint code* that MLflow imports when it starts the
#   model server.  Both voice and image_gen import torch lazily inside method bodies,
#   so by the time any predict() call arrives torch has NOT been imported yet.
#   Preloading the nvidia pip-package shared libraries here — once, at server startup —
#   puts them into the process's global symbol table (RTLD_GLOBAL) before torch's own
#   dlopen() calls happen.  That lets torch's linker resolve CUDA 12.8 symbols from
#   the pip-package libs instead of the older system CUDA at /usr/local/cuda/lib64/.
#
# Why ctypes, not LD_LIBRARY_PATH?
#   glibc caches LD_LIBRARY_PATH at process start.  Modifying os.environ after the
#   process is already running has zero effect on subsequent dlopen() calls.  ctypes
#   with RTLD_GLOBAL is the only way to inject new symbol bindings into a live process.
#
# Load order matters:
#   nvJitLink must load before cusparse (cusparse links against nvJitLink symbols).
#   We achieve deterministic ordering with an explicit priority list and a fallback
#   lexicographic sort for everything else.

_NVIDIA_SITE_PATHS = [
    # nvidia pip packages in the base conda env (confirmed location on AI Studio)
    "/opt/conda/lib/python3.12/site-packages/nvidia",
    # same packages if installed inside the aistudio venv
    "/opt/conda/envs/aistudio/lib/python3.12/site-packages/nvidia",
]

# Lower index = loaded first.  Libraries that others depend on must appear earlier.
_LOAD_ORDER_PRIORITY = [
    "nvjitlink",  # nvJitLink — required by cusparse
    "cublas",  # cuBLAS — required by many torch ops
    "cusparse",  # cuSPARSE — requires nvJitLink
    "cusparselt",  # cuSPARSELt — requires cusparse
    "cudnn",  # cuDNN — requires cublas
    "cufft",  # cuFFT
    "curand",  # cuRAND
    "cusolver",  # cuSOLVER
    "nccl",  # NCCL (multi-GPU comms)
]


def _lib_sort_key(path: str) -> tuple:
    """Return (priority_index, path) so foundational libs sort before dependents."""
    name = os.path.basename(path).lower()
    for i, token in enumerate(_LOAD_ORDER_PRIORITY):
        if token in name:
            return (i, path)
    return (len(_LOAD_ORDER_PRIORITY), path)


def _preload_nvidia_libs() -> None:
    """
    Preload all nvidia pip-package shared libraries with RTLD_GLOBAL.

    Called once at module import time (server startup).  Silently skips libs that
    cannot be loaded (e.g., wrong architecture or already loaded by another path).
    """
    collected: list[str] = []
    for site_root in _NVIDIA_SITE_PATHS:
        # Each nvidia-*-cu12 package installs .so files under nvidia/<pkg>/lib/
        pattern = os.path.join(site_root, "*", "lib", "*.so*")
        collected.extend(glob.glob(pattern))

    if not collected:
        logger.info(
            "ℹ️ No nvidia pip-package .so files found — "
            "CUDA library preload skipped (expected in non-GPU or CI environments)"
        )
        return

    # Deduplicate while preserving sort order
    seen: set[str] = set()
    unique = [p for p in sorted(set(collected), key=_lib_sort_key) if p not in seen and not seen.add(p)]  # type: ignore[func-returns-value]

    loaded, skipped = 0, 0
    for so_path in unique:
        try:
            ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            loaded += 1
        except OSError:
            skipped += 1

    logger.info(
        "🔧 CUDA preload: %d libs loaded, %d skipped (already mapped or incompatible)",
        loaded,
        skipped,
    )


_preload_nvidia_libs()


# ─────────────────────────────────────────────────────────────────────────────


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
        ├── data/             ← optional: sample documents for document analyzer
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
        logger.info(
            "ℹ️ No secrets.yaml found — running without external API integrations"
        )

    # ── 3. Resolve docs path ──────────────────────────────────────────────────
    docs_path_candidate = data_path / "data"
    docs_path = str(docs_path_candidate) if docs_path_candidate.exists() else None

    # ── 4. Resolve model path ─────────────────────────────────────────────────
    # Priority:
    #   1. models/ inside the artifact (copied by logger.py at register time)
    #   2. MODEL_ARTIFACTS_PATH env var (set by AI Studio serving container)
    #   3. model_path from config.yaml (fallback for local dev / notebook usage)
    capability = config.get("capability", "chatbot")
    artifact_models = os.path.join(data_path, "models")

    if os.path.exists(artifact_models) and os.listdir(artifact_models):
        # Set env var so get_model_path() (and serving containers) can locate files.
        # This matches the pattern used by all non-educational blueprints.
        os.environ["MODEL_ARTIFACTS_PATH"] = artifact_models
        from src.utils import get_model_path

        # The Logger stamps "_artifact_model_keys" into config.yaml at log time,
        # listing exactly which keys it copied into models/.
        artifact_keys = config.get("_artifact_model_keys", [])
        model_path = artifact_models  # default fallback (overwritten below)
        for key in artifact_keys:
            config_val = config.get(key, "")
            if not config_val:
                continue
            resolved = get_model_path(str(config_val))
            if os.path.isfile(resolved) or os.path.isdir(resolved):
                config[key] = resolved
                logger.info(f"✅ {key} resolved from artifact: {resolved}")
                if key == "model_path":
                    model_path = resolved
            else:
                logger.warning(
                    f"⚠️ {key} not found in artifacts ({resolved}). "
                    "Keeping original config value."
                )

        if model_path == artifact_models:
            # model_path key was absent or unresolved — scan for any GGUF as fallback
            gguf_files = sorted(
                f for f in os.listdir(artifact_models) if f.endswith(".gguf")
            )
            if gguf_files:
                model_path = os.path.join(artifact_models, gguf_files[0])
                logger.info(f"✅ model_path resolved by GGUF scan: {model_path}")
    else:
        model_path = os.environ.get(
            "MODEL_ARTIFACTS_PATH", config.get("model_path", "")
        )
        if model_path:
            logger.info(f"ℹ️ Using model_path fallback: {model_path}")
        else:
            logger.warning(
                "⚠️ model_path is not set.\n"
                "   Not found in artifact, MODEL_ARTIFACTS_PATH, or config.yaml.\n"
                "   Model will be unavailable until this is configured."
            )

    # ── 5. Select the right Model class based on config["capability"] ────────
    # Each config YAML has a "capability" key that tells loader.py which Model
    # class to instantiate. This is what enables per-capability model registration.
    if capability == "image_gen":
        from src.mlflow.models.image_gen import ImageGenModel as Model
    elif capability == "document":
        from src.mlflow.models.document import DocumentModel as Model
    elif capability == "voice":
        from src.mlflow.models.voice import VoiceModel as Model
    else:
        # Default: chatbot (covers "chatbot" and any unrecognized capability string)
        from src.mlflow.models.chatbot import ChatbotModel as Model

    logger.info(
        f"Building {Model.__name__}: model_path={model_path!r}, docs_path={docs_path!r}"
    )

    # ── 6. Instantiate and return the Model ──────────────────────────────────
    # All Model classes share the same constructor signature for loader compatibility.
    return Model(
        config=config,
        docs_path=docs_path,
        model_path=model_path,
        secrets=secrets,
    )
