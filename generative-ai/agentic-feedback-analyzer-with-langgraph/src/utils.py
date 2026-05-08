# ─────── Standard Library Imports ───────
import base64  # Encoding and decoding binary data
import logging  # Logging utilities
import os  # Operating system interface
import sys  # System-specific parameters and functions
import time  # Time-related utilities
from functools import wraps  # Function decorators support
from typing import Dict, Any  # Type annotations

# ─────── Third-Party Package Imports ───────
from IPython.display import (
    HTML,
    display,
)  # Rich HTML display utilities for Jupyter environments

# Color and emoji mapping per level
STYLE_MAP = {
    logging.DEBUG: {"bg": "#1e90ff", "fg": "white", "icon": "🔍"},
    logging.INFO: {"bg": "#228B22", "fg": "white", "icon": "✅"},
    logging.WARNING: {"bg": "#ffcc00", "fg": "black", "icon": "⚠️"},
    logging.ERROR: {"bg": "#cc0000", "fg": "white", "icon": "❌"},
    logging.CRITICAL: {"bg": "#8B0000", "fg": "white", "icon": "🔥"},
}


class EmojiStyledJupyterHandler(logging.Handler):
    def emit(self, record):
        style = STYLE_MAP.get(
            record.levelno, {"bg": "white", "fg": "black", "icon": "💬"}
        )
        formatted = self.format(record)
        html = f"""
        <div style="background-color: {style['bg']}; color: {style['fg']};
                    padding: 4px 8px; font-family: monospace; border-radius: 4px;">
            {style["icon"]} {formatted}
        </div>
        """
        display(HTML(html))


# Logger setup
logger = logging.getLogger("AIS_logger")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()

formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

handler = EmojiStyledJupyterHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


def log_timing(func):
    """
    Decorator that logs the execution time of a function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(
            f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds."
        )
        return result

    return wrapper


def get_response_from_llm(llm, system_prompt, user_prompt):
    meta_llama_prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    return llm.invoke(meta_llama_prompt)


def display_image(image_bytes: bytes, width: int = 400) -> str:
    """
    Converts image bytes to an HTML string for visualization in Jupyter.

    Args:
        image_bytes (bytes): Raw image content in PNG/JPEG format.
        width (int): Desired width in pixels for display.

    Returns:
        str: HTML <img> tag with base64 image data.
    """
    decoded_img_bytes = base64.b64encode(image_bytes).decode("utf-8")
    html = f'<img src="data:image/png;base64,{decoded_img_bytes}" style="width: {width}px;" />'
    display(HTML(html))


def json_schema_from_type(input_type: type):
    """
    Convert a Python type to a basic JSON schema representation.
    Used for MLflow input/output signatures.
    """
    mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }
    return mapping.get(input_type, {"type": "string"})


def get_model_path(model_name: str) -> str:
    """
    Get the full path to the model file using the artifacts path and model name.

    Args:
        model_name: Name of the model file or full path (will extract filename)

    Returns:
        Full path to the model file
    """
    # Extract just the filename if model_name contains a path
    filename = os.path.basename(model_name)

    artifacts_path = os.environ.get("MODEL_ARTIFACTS_PATH", "")
    model_path = os.path.join(artifacts_path, filename)

    return model_path


def load_config(config_path: str = "../configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file with proper error handling.

    Args:
        config_path: Path to the configuration YAML file

    Returns:
        Dictionary containing the project configurations, empty dict if file not found

    Raises:
        FileNotFoundError: If the config file is not found in strict mode
    """
    import yaml

    # Convert to absolute path for consistency
    config_path = os.path.abspath(config_path)

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Configuration loaded from: {config_path}")
            return config if config is not None else {}
        except yaml.YAMLError as e:
            logger.error(f"❌ Failed to parse YAML config at {config_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"❌ Failed to load config from {config_path}: {e}")
            return {}
    else:
        logger.warning(f"⚠️ Config file not found at: {config_path}, using defaults")
        return {}
