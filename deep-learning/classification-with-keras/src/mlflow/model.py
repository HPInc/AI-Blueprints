"""
Business logic Model for classification-with-keras (MNIST).
This class contains the model loading and predict logic extracted from the notebook.
No MLflow dependencies here.
"""

import os
import base64
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, config: Dict[str, Any], docs_path: str, model_path: Optional[str] = None, secrets: Optional[Dict[str, Any]] = None):
        self.config = config
        self.docs_path = docs_path
        self.model_path = model_path
        self.secrets = secrets

        # Keras model placeholder
        self.model = None

        # Setup environment
        self._setup_environment()
        # Load Keras model if provided
        if self.model_path and os.path.exists(self.model_path):
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model(self.model_path)
                logger.info(f"Keras model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load Keras model from {self.model_path}: {e}")
                raise
        else:
            logger.info("No Keras model_path provided or file does not exist; model remains uninitialized")

    def _setup_environment(self):
        # Load secrets to env if provided
        if self.secrets:
            for k, v in self.secrets.items():
                os.environ[k] = str(v)

    def _base64_to_numpy(self, base64_string: str) -> np.ndarray:
        import base64 as _b64
        from PIL import Image
        from io import BytesIO

        image_data = _b64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        if image.mode != 'L':
            image = image.convert('L')
        if image.size != (28, 28):
            image = image.resize((28, 28))
        numpy_array = np.array(image).astype('float32') / 255.0
        numpy_array = numpy_array.reshape(1, 28, 28, 1)
        return numpy_array

    def predict(self, model_input, params=None):
        try:
            if isinstance(model_input, pd.DataFrame):
                image_input = model_input.iloc[0, 0]
            elif isinstance(model_input, list):
                image_input = model_input[0]
            else:
                image_input = str(model_input)

            base64_array = self._base64_to_numpy(image_input)

            if self.model is None:
                raise RuntimeError("Keras model is not loaded; cannot perform predictions")

            predictions = self.model.predict(base64_array)
            predicted_classes = np.argmax(predictions, axis=1)
            return predicted_classes.tolist()
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
