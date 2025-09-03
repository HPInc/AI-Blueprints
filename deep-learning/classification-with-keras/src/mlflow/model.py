"""
Standalone Model class for MNIST handwritten digit classification.

Business Logic Layer
- Handles image classification using TensorFlow/Keras CNN model
- Manages model initialization and prediction logic for MNIST digits
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

import os
import sys
import base64
import logging
from io import BytesIO
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from PIL import Image

# TensorFlow imports
import tensorflow as tf

# Add the src directory to the path to import utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils import load_config

# Set up logger
logger = logging.getLogger(__name__)


class Model:
    """
    Standalone model class with no MLflow inheritance.
    Handles MNIST digit classification using TensorFlow/Keras.
    """
    
    def __init__(self, config: dict, model_path: str = None):
        """
        Initialize the MNIST classification model.
        
        Args:
            config: Configuration dictionary containing model settings
            model_path: Path to the trained Keras model file
        """
        self.config = config
        self.model_path = model_path
        self.model = None
        
        # Load the model
        self._load_model()
        
        logger.info("✅ MNIST Model initialized successfully")
    
    def _load_model(self):
        """
        Load the trained TensorFlow/Keras model.
        """
        try:
            if self.model_path and os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"✅ Model loaded from: {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            raise
    
    def predict(self, model_input, params=None):
        """
        Predict the handwritten digit from base64 encoded image input.
        
        Args:
            model_input: Input data containing base64 encoded image
            params: Additional parameters (optional)
            
        Returns:
            List containing predicted digit class
        """
        try:
            # Handle different input formats
            if isinstance(model_input, pd.DataFrame):
                image_input = model_input.iloc[0, 0]
            elif isinstance(model_input, dict) and "digit" in model_input:
                image_input = model_input["digit"]
            elif isinstance(model_input, list):
                image_input = model_input[0]
            else:
                image_input = str(model_input)
            
            # Convert base64 to numpy array
            base64_array = self._base64_to_numpy(image_input)
            
            # Make prediction
            predictions = self.model.predict(base64_array)
            predicted_classes = np.argmax(predictions, axis=1)
            
            return predicted_classes.tolist()
        
        except Exception as e:
            logger.error(f"❌ Error performing prediction: {str(e)}")
            raise
    
    def _base64_to_numpy(self, base64_string):
        """
        Convert base64 string to numpy array for MNIST digit prediction.
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            Numpy array formatted for MNIST model input (28x28x1)
        """
        try:
            # Decode base64 string to image
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
            
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Resize to 28x28 if needed
            if image.size != (28, 28):
                image = image.resize((28, 28))
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32)
            image_array = image_array / 255.0
            
            # Reshape to match model input format (1, 28, 28, 1)
            image_array = image_array.reshape(1, 28, 28, 1)
            
            return image_array
            
        except Exception as e:
            logger.error(f"❌ Error converting base64 to numpy array: {str(e)}")
            raise
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if self.model:
            return {
                "model_type": "TensorFlow/Keras CNN",
                "input_shape": self.model.input_shape,
                "output_shape": self.model.output_shape,
                "model_path": self.model_path,
                "num_classes": 10
            }
        else:
            return {"status": "Model not loaded"}