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


# 0) Quiet noisy backends early (must happen before importing TF/JAX/etc.)
os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL", "2"
)  # 0=all, 1=hide INFO, 2=hide INFO+WARN, 3=hide all

# 1) Reset any pre-installed handlers (IPython adds one to the root logger)
root = logging.getLogger()
for h in root.handlers[:]:
    root.removeHandler(h)
root.setLevel(logging.WARNING)  # keep root quiet; our app logger will control verbosity

# 2) Tame absl/glog so it doesn't emit "Logging before flag parsing..." or mirror to stderr
try:
    from absl import flags, logging as absl_logging

    # Pretend-parse flags in notebooks so absl doesn't complain
    if not flags.FLAGS.is_parsed():
        # Use only argv[0]; avoids errors from unknown notebook args
        flags.FLAGS(sys.argv[:1])

    # Reduce absl verbosity; set to FATAL to clamp harder if needed
    absl_logging.set_verbosity(absl_logging.ERROR)
    # Prevent info/warn from going to stderr
    try:
        absl_logging.set_stderrthreshold("error")
    except Exception:
        pass
except Exception:
    # absl not present — nothing to do
    pass

# 3) (Optional) Calm down Transformers/HF logging if present
try:
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
except Exception:
    pass

# 4) Forward Python warnings through logging (so everything is centralized)
logging.captureWarnings(True)

# 5) Application logger with a single handler and no propagation (prevents duplicates)
logger = logging.getLogger("AIS_logger")
logger.setLevel(logging.DEBUG)  # adjust app verbosity here
logger.propagate = False  # critical: do NOT bubble up to root (avoids duplicate lines)
logger.handlers.clear()

# If you have a custom Jupyter emoji handler, use it; otherwise, fallback to StreamHandler
try:
    handler = (
        EmojiStyledJupyterHandler()
    )  # user-defined handler available in your codebase
except NameError:
    handler = logging.StreamHandler()

formatter = logging.Formatter(
    fmt="✅ %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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


def load_secrets_to_env(secrets_path: str) -> None:
    """
    Load secrets from YAML file into environment variables.

    Args:
        secrets_path: Path to the secrets YAML file.
    """
    if os.path.exists(secrets_path):
        with open(secrets_path, "r") as file:
            secrets = yaml.safe_load(file)

        if secrets:
            for key, value in secrets.items():
                os.environ[key] = str(value)


def load_secrets(secrets_path: str = None) -> Dict[str, Any]:
    """
    Load secrets from environment variables or YAML file.

    Args:
        secrets_path: Path to the secrets YAML file (optional).

    Returns:
        Dictionary containing secrets.
    """
    secrets = {}

    # Load from environment variables that might contain secrets
    for key, value in os.environ.items():
        if any(
            secret_key in key.lower()
            for secret_key in ["key", "token", "secret", "password"]
        ):
            secrets[key] = value

    # Optionally load from file if provided
    if secrets_path and os.path.exists(secrets_path):
        with open(secrets_path, "r") as file:
            file_secrets = yaml.safe_load(file)
        if file_secrets:
            secrets.update(file_secrets)

    return secrets
