"""
Logger for MLflow model registration (models-from-code) - classification-with-keras
"""

import os
import logging
import shutil
import yaml
import tempfile

logger = logging.getLogger(__name__)


class Logger:
    @classmethod
    def log_model(cls, signature, artifact_path="AIStudio-Model", config_path="configs/config.yaml", docs_path="data/", secrets_dict=None, model_path=None, demo_folder=None):
        import mlflow

        temp_base = tempfile.gettempdir()
        temp_dir = os.path.join(temp_base, "model_artifacts")

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Copy config
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at: {config_path}")
            shutil.copy2(config_path, os.path.join(temp_dir, "config.yaml"))

            # Copy data/docs
            data_temp_dir = os.path.join(temp_dir, "data")
            os.makedirs(data_temp_dir, exist_ok=True)
            if docs_path and os.path.exists(docs_path):
                for item in os.listdir(docs_path):
                    item_path = os.path.join(docs_path, item)
                    if os.path.isfile(item_path):
                        shutil.copy2(item_path, data_temp_dir)
                    elif os.path.isdir(item_path):
                        shutil.copytree(item_path, os.path.join(data_temp_dir, item))

            # Copy demo
            if demo_folder and os.path.exists(demo_folder):
                shutil.copytree(demo_folder, os.path.join(temp_dir, "demo"))

            # Secrets
            if secrets_dict:
                with open(os.path.join(temp_dir, "secrets.yaml"), 'w') as f:
                    yaml.safe_dump(secrets_dict, f)

            # Models
            if model_path and os.path.exists(model_path):
                models_temp_dir = os.path.join(temp_dir, "models")
                os.makedirs(models_temp_dir, exist_ok=True)
                if os.path.isfile(model_path):
                    shutil.copy2(model_path, os.path.join(models_temp_dir, os.path.basename(model_path)))
                else:
                    shutil.copytree(model_path, models_temp_dir, dirs_exist_ok=True)

                # Attempt ONNX export for copied model files (best-effort).
                # Strategy: for common Keras/TensorFlow models (.keras or SavedModel dirs),
                # load the model and use onnx_utils ModelExportConfig -> conversion helpers
                # to generate ONNX and Triton-style directories inside the temp models folder.
                try:
                    from onnx_utils import ModelExportConfig, _generate_onnx_from_models, _create_model_directories

                    # Build list of candidate model paths inside models_temp_dir
                    candidate_paths = [os.path.join(models_temp_dir, p) for p in os.listdir(models_temp_dir)]
                    model_configs = []

                    for p in candidate_paths:
                        name = os.path.splitext(os.path.basename(p))[0]

                        # Try loading Keras models (file or directory). If tensorflow not available or load fails,
                        # skip this path and continue.
                        try:
                            import tensorflow as tf

                            # tf.keras.models.load_model supports both files and SavedModel dirs
                            loaded = tf.keras.models.load_model(p)
                            cfg = ModelExportConfig(model=loaded, model_name=name, create_triton_structure=True)
                            model_configs.append(cfg)
                        except Exception:
                            # Not a Keras model or failed to load; continue to next candidate
                            continue

                    if model_configs:
                        # Run conversion inside the models_temp_dir so generated dirs live there
                        orig_cwd = os.getcwd()
                        try:
                            os.chdir(models_temp_dir)
                            model_result = _generate_onnx_from_models(model_configs)

                            # Create Triton-style directories for generated model dirs
                            if model_result:
                                _create_model_directories(model_dirs=model_result, create_triton_structure=True)
                                logger.info("ONNX model directories created inside temp models folder")
                        finally:
                            os.chdir(orig_cwd)

                except Exception as e:
                    # ONNX conversion is a best-effort step; log and continue
                    logger.warning(f"ONNX export step skipped or failed: {e}")

            mlflow.pyfunc.log_model(
                name=artifact_path,
                loader_module="src.mlflow.loader",
                data_path=temp_dir,
                code_paths=["../src"],
                signature=signature,
                pip_requirements="../requirements.txt"
            )
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
