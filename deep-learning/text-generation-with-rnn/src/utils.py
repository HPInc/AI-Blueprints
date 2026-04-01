# -----------------------------
# Standard library imports
# -----------------------------
import base64  # Encoding and decoding binary data
import logging  # Logging utilities
import os  # Operating system utilities (paths, env vars, files)
import shutil  # High-level file operations (copy, move, delete)
import sys  # Python runtime environment manipulation
import tempfile  # Temporary file and directory management
import time  # Time-related utilities
from functools import wraps  # Function decorator support
from pathlib import Path  # Object-oriented filesystem paths
from typing import Any, Dict, Optional, Tuple  # Type hints for annotations
from urllib.parse import urlparse  # URL parsing utilities

# -----------------------------
# Third-party imports
# -----------------------------
import boto3  # AWS SDK for Python
import yaml  # YAML parsing and serialization
from botocore import UNSIGNED  # Disable request signing
from botocore.config import Config  # Botocore configuration
from IPython.display import HTML, display  # Rich HTML display utilities (Jupyter)

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


def load_config(config_path: str = "../configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file is not found.
    """
    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml file not found in path: {config_path}")

    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config


def configure_proxy(config: Dict[str, Any]) -> None:
    """
    Configure proxy settings based on provided configuration.

    Args:
        config: Configuration dictionary that may contain a "proxy" key.
    """
    if "proxy" in config and config["proxy"]:
        os.environ["HTTPS_PROXY"] = config["proxy"]


def get_ui_mode(config: Dict[str, Any]) -> str:
    """
    Get the UI mode from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        UI mode string (static, streamlit, or gradio).
    """
    return config.get("ui", {}).get("mode", "static")


def get_service_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get service configuration settings.

    Args:
        config: Configuration dictionary.

    Returns:
        Service configuration dictionary.
    """
    return config.get("service", {})


def get_ports_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get port configuration settings.

    Args:
        config: Configuration dictionary.

    Returns:
        Ports configuration dictionary.
    """
    return config.get("ports", {})


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


def download_from_s3_uri(s3_uri: str, local_path: str | Path) -> str:
    """
    Download an S3 object (s3://bucket/key) into a local directory or file path.

    Args:
        s3_uri: Full S3 URI (e.g. s3://bucket/path/to/file.txt)
        local_path: Either a directory (file will be saved under its basename)
                    or a full file path (file will be saved exactly there).

    Returns:
        Absolute path of the downloaded file.
    """
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    basename = os.path.basename(key) or "downloaded_file"

    local_path = Path(local_path).expanduser().resolve()
    if local_path.exists() and local_path.is_dir():
        target = local_path / basename
    elif local_path.suffix:  # looks like a file path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        target = local_path
    else:
        local_path.mkdir(parents=True, exist_ok=True)
        target = local_path / basename

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        s3.download_file(bucket, key, str(tmp_path))
        shutil.move(str(tmp_path), str(target))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return str(target)
