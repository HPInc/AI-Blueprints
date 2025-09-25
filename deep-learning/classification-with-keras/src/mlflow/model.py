"""
Standalone Model class for MNIST digit classification.

Business Logic Layer
- Handles handwritten digit classification using Keras/TensorFlow models
- Manages model initialization, base64 image conversion, and prediction logic
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

import os
import base64
import logging
from typing import Dict, Any, Optional, Union, List
from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image

# Set up logger
logger = logging.getLogger(__name__)


class Model:
    """
    Standalone model class with no MLflow inheritance.
    Handles MNIST digit classification with base64 image input.
    """

    def __init__(
        self, config: dict, docs_path: str, secrets: dict = None, model_path: str = None
    ):
        """
        Initialize the MNIST classification model.

        Args:
            config: Configuration dictionary containing model settings
            docs_path: Path to data directory (required by generic loader, may contain model files)
            secrets: Secrets dictionary (optional, not used in MNIST classification)
            model_path: Path to model file (optional, can specify specific model location)
        """
        self.config = config
        self.docs_path = docs_path
        self.secrets = secrets
        self.model_path = model_path

        # Load the Keras model from artifacts
        self.model = self._load_keras_model()

        logger.info("✅ MNIST Model initialized successfully")

    def _load_keras_model(self):
        """
        Load the Keras model from artifacts.

        Returns:
            Loaded Keras model ready for prediction
        """
        try:
            import tensorflow as tf

            # Try to find the model file in the docs_path (artifacts directory)
            # Look for common Keras model file extensions
            possible_filenames = [
                "mnist_model.keras",
                "model_keras_mnist.keras",
                "model.keras",
                "mnist_model.h5",
                "model.h5",
            ]

            model_file_path = None

            # First, check if a specific model_path was provided
            if self.model_path and os.path.exists(self.model_path):
                model_file_path = self.model_path
                logger.info(f"Using specified model path: {model_file_path}")
            else:
                # Search for model files in the docs_path (artifacts directory)
                for filename in possible_filenames:
                    potential_path = os.path.join(self.docs_path, filename)
                    if os.path.exists(potential_path):
                        model_file_path = potential_path
                        logger.info(f"Found model file: {model_file_path}")
                        break

                # Also check in parent directory of docs_path (artifacts root)
                if not model_file_path:
                    artifacts_root = os.path.dirname(self.docs_path)
                    for filename in possible_filenames:
                        potential_path = os.path.join(artifacts_root, filename)
                        if os.path.exists(potential_path):
                            model_file_path = potential_path
                            logger.info(
                                f"Found model file in artifacts root: {model_file_path}"
                            )
                            break

            if not model_file_path:
                raise FileNotFoundError(
                    f"No Keras model file found. Searched in {self.docs_path} and parent directory for: {possible_filenames}"
                )

            # Load the Keras model
            keras_model = tf.keras.models.load_model(model_file_path)
            logger.info(f"✅ Keras model loaded successfully from: {model_file_path}")

            return keras_model

        except Exception as e:
            logger.error(f"❌ Error loading Keras model: {str(e)}")
            raise

    def predict(
        self,
        model_input: Union[pd.DataFrame, List, str],
        params: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """
        Predict digit from base64 encoded image.

        Args:
            model_input: Input data containing base64 encoded image
            params: Optional parameters (not used in current implementation)

        Returns:
            List containing predicted digit(s)
        """
        try:
            # Extract base64 image from input
            if isinstance(model_input, pd.DataFrame):
                image_input = model_input.iloc[0, 0]
            elif isinstance(model_input, list):
                image_input = model_input[0] if model_input else ""
            else:
                image_input = str(model_input)

            # Convert base64 to numpy array
            base64_array = self._base64_to_numpy(image_input)

            # Make prediction using the Keras model
            predictions = self.model.predict(base64_array)
            predicted_classes = np.argmax(predictions, axis=1)

            return predicted_classes.tolist()

        except Exception as e:
            logger.error(f"❌ Error performing prediction: {str(e)}")
            raise

    def _base64_to_numpy(self, base64_string: str) -> np.ndarray:
        """
        Convert base64 string to numpy array for MNIST digit prediction.

        Args:
            base64_string: Base64 encoded image string

        Returns:
            Numpy array shaped for model input (1, 28, 28, 1)
        """
        try:
            # Decode the base64 string
            image_data = base64.b64decode(base64_string)

            # Open the image using PIL
            image = Image.open(BytesIO(image_data))

            # Convert to grayscale if not already
            if image.mode != "L":
                image = image.convert("L")

            # Resize to 28x28 if needed
            if image.size != (28, 28):
                image = image.resize((28, 28))

            # Convert to numpy array
            numpy_array = np.array(image)

            # Normalize pixel values to 0-1 range
            numpy_array = numpy_array.astype("float32") / 255.0

            # Reshape for model input (1, 28, 28, 1)
            numpy_array = numpy_array.reshape(1, 28, 28, 1)

            return numpy_array

        except Exception as e:
            logger.error(f"Error converting base64 to numpy: {str(e)}")
            raise
