"""
MLflow Registry Layer — Logger class.

What is this file for?
    The Logger class is responsible for packaging all the pieces of your AI project
    (code, config, models, demo assets) and saving them to the MLflow Model Registry.

    Once logged, you can:
    - Version and compare different model configurations
    - Load the model by name (e.g., `mlflow.pyfunc.load_model("models:/educational-quickstart/1")`)
    - Serve the model as a REST API (`mlflow models serve ...`)
    - Track experiments and compare runs in the MLflow UI

    Learn more about the MLflow Model Registry:
    https://mlflow.org/docs/latest/model-registry.html

v2.0.0 Architecture Summary:
    This is the third (outermost) layer. It has ONE job: packaging and logging.

           loader.py          ← how MLflow reconstructs the model at serving time
           model.py           ← inference logic (framework-agnostic)
    >>>    logger.py          ← how to save the model + artifacts to MLflow (you are here)

    No inference code lives here. No loading code lives in model.py.
    Separation of concerns makes each file simple and testable.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import mlflow

logger = logging.getLogger(__name__)


class Logger:
    """
    Packages and logs the Educational Quickstart model to MLflow.

    Usage (from register-model.ipynb):

        with mlflow.start_run(run_name="educational-quickstart-v1"):
            Logger.log_model(
                signature=signature,
                artifact_path="educational-quickstart",
                config_path="../configs/config.yaml",
                docs_path="../data/input",
                secrets_dict={},
                model_path="/home/jovyan/datafabric/meta-llama3.1-8b-Q8/...",
                demo_folder="../demo/streamlit",
            )
            mlflow.register_model(...)
    """

    @classmethod
    def log_model(
        cls,
        signature,
        artifact_path: str,
        config_path: str,
        docs_path: Optional[str] = None,
        secrets_dict: Optional[Dict] = None,
        model_path: Optional[str] = None,
        demo_folder: Optional[str] = None,
        extra_pip_requirements: Optional[List[str]] = None,
    ) -> None:
        """
        Package all artifacts and call mlflow.pyfunc.log_model().

        This method creates a temporary directory containing everything MLflow
        needs to reconstruct and serve the model later:

            temp_dir/
            ├── config.yaml          ← copied from config_path
            ├── secrets.yaml         ← generated from secrets_dict (if provided)
            ├── docs/                ← copied from docs_path (if provided)
            └── demo/                ← copied from demo_folder (if provided)

        MLflow then reads this directory via loader.py when loading the model.

        Args:
            signature:    MLflow ModelSignature — defines input/output schema.
                          Create with: mlflow.models.infer_signature(sample_input, sample_output)
                          Learn more: https://mlflow.org/docs/latest/models.html#model-signature
            artifact_path: The name under which this model is stored in MLflow runs.
                           This appears as a folder in the MLflow artifacts UI.
            config_path:   Path to configs/config.yaml to embed with the model.
            docs_path:     Optional path to a directory of documents (document analyzer).
            secrets_dict:  Optional dict of secrets to serialize as secrets.yaml.
                           These are written to disk temporarily and never committed to git.
            model_path:    Full path to the .gguf model file used during inference.
                           This is embedded into the logged config so loader.py can find it.
            demo_folder:   Optional path to the Streamlit demo folder (for CSS/logos).
            extra_pip_requirements: Additional pip packages beyond requirements.txt.
        """
        # Create a temporary directory to stage all artifacts
        # It is automatically cleaned up after the with-block, even if an exception occurs
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # ── 1. Copy config.yaml ──────────────────────────────────────────
            config_dest = tmp_path / "config.yaml"
            shutil.copy2(config_path, config_dest)
            logger.info(f"✅ Copied config: {config_path} → {config_dest}")

            # ── 2. Optionally write secrets.yaml ─────────────────────────────
            if secrets_dict:
                import yaml

                secrets_dest = tmp_path / "secrets.yaml"
                with open(secrets_dest, "w") as f:
                    yaml.dump(secrets_dict, f, default_flow_style=False)
                logger.info("✅ Wrote secrets.yaml (contents not logged for security)")

            # ── 3. Optionally copy docs directory ────────────────────────────
            if docs_path and Path(docs_path).exists():
                docs_dest = tmp_path / "docs"
                shutil.copytree(docs_path, docs_dest)
                logger.info(f"✅ Copied docs: {docs_path} → {docs_dest}")

            # ── 4. Optionally copy demo assets ───────────────────────────────
            if demo_folder and Path(demo_folder).exists():
                demo_dest = tmp_path / "demo"
                shutil.copytree(demo_folder, demo_dest)
                logger.info(f"✅ Copied demo: {demo_folder} → {demo_dest}")

            # ── 5. Patch model_path into config ──────────────────────────────
            # If a model_path was explicitly passed, update the config.yaml copy
            # so that loader.py can find the model after MLflow restores artifacts.
            if model_path:
                import yaml

                with open(config_dest, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                cfg["model_path"] = model_path
                with open(config_dest, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False)
                logger.info(f"✅ Patched model_path in config.yaml: {model_path}")

            # ── 6. Build pip requirements list ───────────────────────────────
            pip_reqs = "../requirements.txt"  # Path relative to where mlflow is run

            # ── 7. Log the model to MLflow ────────────────────────────────────
            # This is the key call that actually saves everything.
            # See: https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.log_model
            logger.info(f"📦 Logging model to MLflow artifact path: '{artifact_path}'")
            mlflow.pyfunc.log_model(
                name=artifact_path,  # Name in the MLflow UI
                loader_module="src.mlflow.loader",  # Which module implements _load_pyfunc
                data_path=str(tmp_path),  # The temp dir we built above
                code_paths=["../src"],  # Python source code to bundle
                signature=signature,  # Input/output schema
                pip_requirements=pip_reqs,  # Dependencies
                registered_model_name=None,  # Register separately (see notebook)
            )
            logger.info("✅ Model logged to MLflow successfully")
