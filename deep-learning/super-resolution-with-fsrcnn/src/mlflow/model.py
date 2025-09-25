"""
Standalone Model class for Super Resolution with FSRCNN.

Business Logic Layer
- Handles image super-resolution using FSRCNN neural network
- Manages PyTorch model initialization, image preprocessing, and postprocessing
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

import os
import sys
import base64
import logging
import warnings
from io import BytesIO
from typing import Dict, Any, Optional
import yaml
import torch
from torch import nn
import numpy as np
from PIL import Image
import pandas as pd

# Add the src directory to the path to import utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Set up logger
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FSRCNN(nn.Module):
    """
    A class for building the architecture of the FSRCNN model.
    """

    def __init__(self, scale_factor):
        """
        Initializes the FSRCNN.
        Args:
            scale_factor: The factor by which to upscale the image.
        """
        try:
            super(FSRCNN, self).__init__()
            self.scale_factor = scale_factor

            self.feature_extraction = nn.Sequential(
                nn.Conv2d(3, 56, kernel_size=5, padding=2), nn.PReLU()
            )
            self.shrinking = nn.Sequential(nn.Conv2d(56, 12, kernel_size=1), nn.PReLU())
            self.non_linear_mapping = nn.Sequential(
                nn.Conv2d(12, 12, kernel_size=3, padding=1),
                nn.PReLU(),
                nn.Conv2d(12, 12, kernel_size=3, padding=1),
                nn.PReLU(),
                nn.Conv2d(12, 12, kernel_size=3, padding=1),
                nn.PReLU(),
                nn.Conv2d(12, 12, kernel_size=3, padding=1),
                nn.PReLU(),
                nn.Conv2d(12, 12, kernel_size=3, padding=1),
                nn.PReLU(),
            )
            self.expanding = nn.Sequential(nn.Conv2d(12, 56, kernel_size=1), nn.PReLU())
            self.deconvolution = nn.ConvTranspose2d(
                56,
                3,
                kernel_size=9,
                stride=scale_factor,
                padding=4,
                output_padding=scale_factor - 1,
            )

            logger.info("FSRCNN initialization done successfully")

        except Exception as e:
            logger.error(f"Error initializing FSRCNN: {str(e)}")
            raise

    def forward(self, x):
        """
        Implementation of the FSRCNN logic, in which, the input passes through every step of the arquiteture.
        Args:
            x: Low resolution image.

        Returns:
            x: High resolution image.
        """
        try:
            x = self.feature_extraction(x)
            x = self.shrinking(x)
            x = self.non_linear_mapping(x)
            x = self.expanding(x)
            x = self.deconvolution(x)
            return x
        except Exception as e:
            logger.error(f"Error implementing FSRCNN logic: {str(e)}")
            raise


class Model:
    """
    Standalone model class with no MLflow inheritance.
    Handles image super-resolution using FSRCNN neural network.
    """

    def __init__(self, config: dict, model_path: str = None):
        """
        Initialize the Model with configuration and model path.

        Args:
            config: Model configuration dictionary
            model_path: Path to the PyTorch model file
        """
        self.config = config
        self.model_path = model_path
        self.model = None

        # Initialize components
        try:
            self._load_model()
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Model: {str(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Model initialization failed: {str(e)}") from e

    def _load_model(self) -> None:
        """Load the FSRCNN PyTorch model."""
        try:
            # Initialize FSRCNN model with scale factor 4
            self.model = FSRCNN(4)
            self.model.to(device)

            # Load model weights if model_path is provided
            if self.model_path and os.path.exists(self.model_path):
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=device)
                )
                logger.info(f"Model weights loaded from: {self.model_path}")
            else:
                logger.warning(
                    f"Model path not found or not provided: {self.model_path}"
                )

            # Set model to evaluation mode
            self.model.eval()
            logger.info("FSRCNN model loaded and set to evaluation mode")

        except Exception as e:
            logger.error(f"Failed to load FSRCNN model: {str(e)}")
            raise

    def preprocess_image(self, base64_str: str) -> torch.Tensor:
        """
        Decode base64 string to image tensor.

        Args:
            base64_str: Base64 encoded image string

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            img_data = base64.b64decode(base64_str)
            img = Image.open(BytesIO(img_data)).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            return img_tensor
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def postprocess_image(self, output_tensor: torch.Tensor) -> str:
        """
        Convert output tensor to base64 string.

        Args:
            output_tensor: Model output tensor

        Returns:
            str: Base64 encoded output image
        """
        try:
            output_np = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            output_img = (output_np * 255).clip(0, 255).astype(np.uint8)
            output_pil = Image.fromarray(output_img)
            buffer = BytesIO()
            output_pil.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error postprocessing image: {str(e)}")
            raise

    def predict(self, model_input, params=None):
        """
        Process input images and generate super-resolution outputs.
        Must return pandas.DataFrame matching original signature.

        Args:
            model_input: Input data containing base64 encoded images
            params: Optional parameters (unused in this implementation)

        Returns:
            List of base64 encoded super-resolution images
        """
        try:
            if not self.model:
                raise RuntimeError("Model not initialized")

            results = []

            # Process each image in the input
            for base64_str in model_input["image"]:
                # Preprocess the input image
                input_tensor = self.preprocess_image(base64_str).to(device)

                # Perform super-resolution
                with torch.no_grad():
                    output_tensor = self.model(input_tensor)

                # Postprocess the output
                output_base64 = self.postprocess_image(output_tensor)
                results.append(output_base64)

            logger.info(f"Successfully processed {len(results)} images")
            return results

        except Exception as e:
            logger.error(f"❌ Error performing prediction: {str(e)}")
            raise
