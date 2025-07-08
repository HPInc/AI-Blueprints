from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf

# Force TensorFlow to use only the CPU (if necessary)
tf.config.set_visible_devices([], 'GPU')

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
        print(f"[load_image] Image loaded with shape: {img_np.shape}, dtype: {img_np.dtype}")
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
        print(f"[convert_to_grayscale] Image converted to grayscale. Shape: {gray.shape}")
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
    print(f"[calculate_complexity] Complexidade calculada (std do espectro): {complexity}")
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

def calculate_image_entropy(image_path: str = None) -> float:
    """
    Calculate the entropy of an image using the global CUSTOM_IMAGE_PATH if no path provided.
    
    Args:
        image_path: Path to the image file. If None, uses CUSTOM_IMAGE_PATH.
        
    Returns:
        float: Entropy value of the image
    """
    if image_path is None:
        image_path = CUSTOM_IMAGE_PATH
        
    print(f"[calculate_image_entropy] Using image_path: {image_path}")
    try:
        metrics = ImageMetrics(image_path)
        entropy = metrics.calculate_entropy()
        print(f"[calculate_image_entropy] Entropy: {entropy}")
        return entropy
    except Exception as e:
        print(f"[calculate_image_entropy] Error: {e}")
        return 0.0

def aggregate_entropy_scores(scores: List[float]) -> Dict[str, float]:
    """Aggregate entropy scores and return summary statistics."""
    total = sum(scores)
    avg = total / len(scores) if scores else 0.0
    return {"Total Entropy": total, "Average Entropy": avg, "Count": len(scores)}

# --- Executor and Aggregator for Complexity ---

def calculate_image_complexity(image_path: str = None) -> float:
    """
    Calculate the complexity of an image using the global CUSTOM_IMAGE_PATH if no path provided.
    
    Args:
        image_path: Path to the image file. If None, uses CUSTOM_IMAGE_PATH.
        
    Returns:
        float: Complexity value of the image
    """
    if image_path is None:
        image_path = CUSTOM_IMAGE_PATH
        
    print(f"[calculate_image_complexity] Using image_path: {image_path}")
    try:
        metrics = ImageMetrics(image_path)
        complexity = metrics.calculate_complexity()
        print(f"[calculate_image_complexity] Complexity: {complexity}")
        return complexity
    except Exception as e:
        print(f"[calculate_image_complexity] Error: {e}")
        return 0.0

def aggregate_complexity_scores(scores: List[float]) -> Dict[str, float]:
    """Aggregate complexity scores and return summary statistics."""
    total = sum(scores)
    avg = total / len(scores) if scores else 0.0
    return {"Total Complexity": total, "Average Complexity": avg, "Count": len(scores)}

# --- Legacy compatibility functions for existing usage ---

def executor_entropy(row=None, **kwargs) -> float:
    """
    Legacy compatibility function for entropy calculation.
    Maintains API compatibility while removing promptquality dependency.
    """
    return calculate_image_entropy()

def aggregator_entropy(scores: List[float], indices: List[int] = None) -> Dict[str, float]:
    """Legacy compatibility function for entropy aggregation."""
    return aggregate_entropy_scores(scores)

def executor_complexity(row=None, **kwargs) -> float:
    """
    Legacy compatibility function for complexity calculation.
    Maintains API compatibility while removing promptquality dependency.
    """
    return calculate_image_complexity()

def aggregator_complexity(scores: List[float], indices: List[int] = None) -> Dict[str, float]:
    """Legacy compatibility function for complexity aggregation."""
    return aggregate_complexity_scores(scores)

# --- Standalone Metric Objects for compatibility ---

class StandaloneScorer:
    """
    Standalone scorer class to replace promptquality CustomScorer dependency.
    Provides basic functionality for image quality metric evaluation.
    """
    def __init__(self, name: str, executor_func, aggregator_func):
        self.name = name
        self.executor = executor_func
        self.aggregator = aggregator_func
    
    def score(self, image_path: str = None) -> float:
        """Calculate score for a single image."""
        if image_path:
            # Temporarily set the global path for compatibility
            original_path = CUSTOM_IMAGE_PATH
            set_custom_image_path(image_path)
            result = self.executor()
            set_custom_image_path(original_path)
            return result
        else:
            return self.executor()
    
    def batch_score(self, image_paths: List[str]) -> Dict[str, float]:
        """Calculate scores for multiple images and return aggregated results."""
        scores = []
        for path in image_paths:
            scores.append(self.score(path))
        return self.aggregator(scores)

# Create scorer instances for backward compatibility
entropy_scorer = StandaloneScorer(
    name="Image Entropy",
    executor_func=executor_entropy,
    aggregator_func=aggregator_entropy
)

complexity_scorer = StandaloneScorer(
    name="Image Complexity",
    executor_func=executor_complexity,
    aggregator_func=aggregator_complexity
)
