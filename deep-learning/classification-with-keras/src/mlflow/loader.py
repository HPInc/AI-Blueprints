"""
MLflow models-from-code loader module for classification-with-keras.
"""

import os
import logging

logger = logging.getLogger(__name__)


def _load_pyfunc(data_path: str):
    """Load the Model class from artifacts and return an instance ready for predict.
    """
    from src.mlflow.model import Model
    from src.utils import load_config

    logger.info(f"Loading Model from artifacts at: {data_path}")

    config_path = os.path.join(data_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    config = load_config(config_path)

    # Load secrets if available
    secrets_path = os.path.join(data_path, "secrets.yaml")
    if os.path.exists(secrets_path):
        from src.utils import load_secrets_to_env, load_secrets
        load_secrets_to_env(secrets_path)
        secrets = load_secrets()
    else:
        secrets = None

    docs_path = os.path.join(data_path, "data")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Documents directory not found at: {docs_path}")

    model_path = config.get("model_path")
    if model_path:
        from src.utils import get_model_path
        models_artifacts_path = os.path.join(data_path, "models")
        os.environ["MODEL_ARTIFACTS_PATH"] = models_artifacts_path
        resolved_model_path = get_model_path(model_path)
        model_path = resolved_model_path

    model = Model(config=config, docs_path=docs_path, model_path=model_path, secrets=secrets)
    return model
