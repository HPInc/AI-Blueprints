from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf

# Force TensorFlow to use only the CPU (if necessary)
tf.config.set_visible_devices([], "GPU")

# Global variable for custom path; default value
CUSTOM_IMAGE_PATH: Optional[str] = "./img_test.jpg"


def set_custom_image_path(path: str) -> None:
    """
    Updates the image path that will be used by executors.
    """
    global CUSTOM_IMAGE_PATH
    CUSTOM_IMAGE_PATH = path
    print(f"[set_custom_image_path] CUSTOM_IMAGE_PATH: {CUSTOM_IMAGE_PATH}")


# --- Auxiliary functions ---


def load_image(image_path: str) -> np.ndarray:
    """
    Loads the image from the given path and converts it to float32 in the range [0, 1].
    """
    try:
        print(f"[load_image] Loading the image of: {image_path}")
        image_data = tf.io.read_file(image_path)
        image = tf.image.decode_image(image_data, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        img_np = image.numpy()
        print(
            f"[load_image] Image loaded with shape: {img_np.shape}, dtype: {img_np.dtype}"
        )
        return img_np
    except Exception as e:
        print(f"[load_image] Error loading image: {e}")
        raise ValueError(f"Error loading image:: {e}")


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to grayscale using the luminance formula.
    """
    if image.ndim == 3 and image.shape[-1] == 3:
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        print(
            f"[convert_to_grayscale] Image converted to grayscale. Shape: {gray.shape}"
        )
        return gray
    print("[convert_to_grayscale] Image is already in grayscale.")
    return image


def shannon_entropy_np(image: np.ndarray, bins: int = 256) -> float:
    """
    Calculates the Shannon entropy for a normalized image in the interval [0, 1].
    """
    hist, _ = np.histogram(image, bins=bins, range=(0, 1), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    print(f"[shannon_entropy_np] Entropia calculada: {entropy}")
    return entropy


def calculate_complexity(image: np.ndarray) -> float:
    """
    Computes a complexity metric based on the standard deviation of the magnitude spectrum
    of the Fourier transform.
    """
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    complexity = np.std(magnitude_spectrum)
    print(
        f"[calculate_complexity] Complexidade calculada (std do espectro): {complexity}"
    )
    return complexity


# --- Class that calculates metrics ---


class ImageMetrics:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = load_image(image_path)

    def calculate_entropy(self) -> float:
        gray = convert_to_grayscale(self.image)
        return shannon_entropy_np(gray)

    def calculate_complexity(self) -> float:
        gray = convert_to_grayscale(self.image)
        return calculate_complexity(gray)


# --- Image Metric Scorer Functions ---


def entropy_scorer(image_path: str = None) -> float:
    """
    Calculate the entropy of an image.

    Args:
        image_path: Path to the image file. If None, uses CUSTOM_IMAGE_PATH.

    Returns:
        float: Entropy value of the image
    """
    path = image_path if image_path is not None else CUSTOM_IMAGE_PATH
    try:
        metrics = ImageMetrics(path)
        return metrics.calculate_entropy()
    except Exception as e:
        print(f"[entropy_scorer] Error calculating entropy: {e}")
        return 0.0


def complexity_scorer(image_path: str = None) -> float:
    """
    Calculate the complexity of an image.

    Args:
        image_path: Path to the image file. If None, uses CUSTOM_IMAGE_PATH.

    Returns:
        float: Complexity value of the image
    """
    path = image_path if image_path is not None else CUSTOM_IMAGE_PATH
    try:
        metrics = ImageMetrics(path)
        return metrics.calculate_complexity()
    except Exception as e:
        print(f"[complexity_scorer] Error calculating complexity: {e}")
        return 0.0
