# ─────── Standard Library Imports ───────
import base64  # Encoding binary data (images) as text for display in browsers
import logging  # Python's built-in system for printing status messages with severity levels
import os  # Interacting with the operating system (file paths, environment variables)
import subprocess  # Running external commands like pip install
import sys  # System-level utilities (Python version, path management)
import time  # Measuring how long things take
from functools import (
    wraps,
)  # Helper for writing decorators (functions that wrap other functions)
from typing import (
    Dict,
    Any,
    Tuple,
)  # Type hints — help editors and readers understand what a function expects

# ─────── Third-Party Package Imports ───────
from IPython.display import (
    HTML,
    display,
)  # Rich HTML display utilities — lets us show styled text and images inside Jupyter notebooks


# ─────── Log Level Styles ────────────────────────────────────────────────────
# Each log level (DEBUG, INFO, WARNING, etc.) gets its own background color and emoji icon.
# This makes notebook output much easier to read at a glance.
STYLE_MAP = {
    logging.DEBUG: {
        "bg": "#1e90ff",
        "fg": "white",
        "icon": "🔍",
    },  # Blue  — detailed debug info
    logging.INFO: {
        "bg": "#228B22",
        "fg": "white",
        "icon": "✅",
    },  # Green — normal progress messages
    logging.WARNING: {
        "bg": "#ffcc00",
        "fg": "black",
        "icon": "⚠️",
    },  # Yellow — something to be aware of
    logging.ERROR: {
        "bg": "#cc0000",
        "fg": "white",
        "icon": "❌",
    },  # Red   — something went wrong
    logging.CRITICAL: {
        "bg": "#8B0000",
        "fg": "white",
        "icon": "🔥",
    },  # Dark red — serious failure
}


class EmojiStyledJupyterHandler(logging.Handler):
    """
    A custom log handler that displays log messages as styled HTML in Jupyter notebooks.

    Why this exists:
        Plain Python `print()` statements don't show color or severity context.
        This handler wraps every log message in colored HTML so you can
        instantly see whether something is informational (green ✅) or an error (red ❌).

    How it works:
        Python's `logging` module calls `emit()` every time a log message is created.
        Here we override `emit()` to render the message as HTML instead of plain text.

    Learn more about Python logging:
        https://docs.python.org/3/library/logging.html
    """

    def emit(self, record):
        # Look up the color/icon for this log level (default to white/💬 if unknown)
        style = STYLE_MAP.get(
            record.levelno, {"bg": "white", "fg": "black", "icon": "💬"}
        )
        # Format the log record into a string (e.g., "2026-01-01 12:00:00 - INFO - message")
        formatted = self.format(record)
        # Wrap the message in an HTML div with the appropriate colors
        html = f"""
        <div style="background-color: {style['bg']}; color: {style['fg']};
                    padding: 4px 8px; font-family: monospace; border-radius: 4px;">
            {style["icon"]} {formatted}
        </div>
        """
        # Display the HTML in the Jupyter cell output
        display(HTML(html))


# ─────── Module-Level Logger ─────────────────────────────────────────────────
# Create a logger named "AIS_logger" that all modules in this project share.
# The logger collects messages and routes them through the EmojiStyledJupyterHandler.
logger = logging.getLogger("AIS_logger")
logger.setLevel(logging.DEBUG)  # Accept all message levels (DEBUG and above)
logger.handlers.clear()  # Remove any previously attached handlers to avoid duplicates

# Define the timestamp format for log messages
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Attach our custom HTML handler to the logger
handler = EmojiStyledJupyterHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


# ─────── Utility Functions ───────────────────────────────────────────────────


def log_timing(func):
    """
    A decorator that automatically logs how long a function takes to run.

    What is a decorator?
        A decorator is a function that wraps another function to add extra behavior.
        You apply it with the '@' symbol above a function definition.

    Example usage:
        @log_timing
        def train_model():
            ...  # This will now automatically log its runtime

    Learn more about Python decorators:
        https://docs.python.org/3/glossary.html#term-decorator
    """

    @wraps(func)  # Preserve the original function's name and docstring
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Record start timestamp (high-precision)
        result = func(*args, **kwargs)  # Run the original function
        end_time = time.perf_counter()  # Record end timestamp
        logger.info(
            f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds."
        )
        return result

    return wrapper


def get_response_from_llm(llm, system_prompt: str, user_prompt: str) -> str:
    """
    Format and send a prompt to a Meta-Llama LLM and return the response.

    What is a prompt?
        A "prompt" is the text you send to an AI model as input. The model reads
        your prompt and generates a continuation (the "response").

    Why the special formatting (<|begin_of_text|> etc.)?
        Meta-Llama models were trained with specific "special tokens" as delimiters
        between the system instruction, the user message, and the assistant reply.
        Using the correct format improves response quality significantly.

    Args:
        llm: A LlamaCpp model instance (loaded in the notebook)
        system_prompt: The instruction telling the AI how to behave
                       (e.g., "You are a helpful assistant")
        user_prompt: The actual question or input from the user

    Returns:
        The model's text response as a string

    Learn more about prompt engineering:
        https://huggingface.co/learn/nlp-course/chapter1/3
    """
    # Build the complete prompt in Meta-Llama's expected chat format
    meta_llama_prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    return llm(meta_llama_prompt)  # Call the model and return its text output


def display_image(image_bytes: bytes, width: int = 400) -> None:
    """
    Display image bytes as inline HTML in a Jupyter notebook.

    Why base64 encoding?
        Jupyter notebooks are HTML-based. To show an image inline, we need to embed
        the raw image data directly in HTML using base64 encoding — a way of
        converting binary data (bytes) into a text-safe string.

    Args:
        image_bytes: Raw image data in PNG or JPEG format
        width: Display width in pixels (default 400)

    Learn more about base64:
        https://docs.python.org/3/library/base64.html
    """
    # Convert raw bytes to a base64 text string
    decoded_img_bytes = base64.b64encode(image_bytes).decode("utf-8")
    # Build an HTML <img> tag with the image data embedded directly
    html = f'<img src="data:image/png;base64,{decoded_img_bytes}" style="width: {width}px;" />'
    display(HTML(html))  # Render the HTML in the Jupyter cell output


def json_schema_from_type(input_type: type) -> dict:
    """
    Convert a Python type to a basic JSON schema dictionary.

    What is a JSON schema?
        JSON Schema is a standard for describing the shape/structure of data.
        MLflow uses it to document what types of inputs and outputs a model expects.

    Args:
        input_type: A Python type (str, int, float, or bool)

    Returns:
        A dict like {"type": "string"} representing the JSON schema

    Learn more about JSON Schema:
        https://json-schema.org/understanding-json-schema/
    """
    mapping = {
        str: {"type": "string"},  # Text data
        int: {"type": "integer"},  # Whole numbers
        float: {"type": "number"},  # Decimal numbers
        bool: {"type": "boolean"},  # True/False values
    }
    return mapping.get(
        input_type, {"type": "string"}
    )  # Default to string if type unknown


def get_model_path(model_name: str) -> str:
    """
    Resolve the full path to a model file using the MODEL_ARTIFACTS_PATH environment variable.

    Why use an environment variable for paths?
        Model files live in different places depending on whether the model is running
        locally (e.g., in datafabric) or inside an MLflow deployment container.
        Using an environment variable keeps paths flexible without changing code.

    Args:
        model_name: The filename or full path of the model file
                    (e.g., "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")

    Returns:
        The full resolved path to the model file

    Learn more about environment variables in Python:
        https://docs.python.org/3/library/os.html#os.environ
    """
    # If model_name is a full path like "/home/jovyan/datafabric/.../model.gguf",
    # extract just the filename ("model.gguf")
    filename = os.path.basename(model_name)

    # Read the base directory from the environment variable (set by the loader at deploy time)
    artifacts_path = os.environ.get("MODEL_ARTIFACTS_PATH", "")
    # Join the base directory with the filename to get the full path
    model_path = os.path.join(artifacts_path, filename)

    return model_path


def load_config(config_path: str = "../configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration settings from a YAML file.

    What is YAML?
        YAML (YAML Ain't Markup Language) is a human-readable format for configuration files.
        It uses indentation and key: value pairs, making it easy to read and edit.
        Our config.yaml stores model paths, port numbers, and other settings.

    Args:
        config_path: Path to the YAML configuration file
                     (default: "../configs/config.yaml" relative to the notebook)

    Returns:
        A dictionary with all configuration keys and values.
        Returns an empty dict {} if the file is not found (safe fallback).

    Learn more about YAML in Python:
        https://pyyaml.org/wiki/PyYAMLDocumentation
    """
    import yaml  # PyYAML library for parsing YAML files

    # Convert to absolute path so relative path issues don't cause failures
    config_path = os.path.abspath(config_path)

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                # yaml.safe_load parses YAML without executing arbitrary code (secure)
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


def log_asset_status(asset_path_or_assets, asset_name: str = "") -> None:
    """
    Check whether a file or directory exists and log a pass/fail status message.

    Accepts two calling styles:

    1. List of asset dicts (as used in starter notebooks)::

        log_asset_status([
            {"name": "SDXL-Turbo model", "path": image_model_path, "required": True},
            {"name": "Config YAML",      "path": "../configs/image_gen.yaml"},
        ])

    2. Two-argument form for a single asset (backward-compatible)::

        log_asset_status("/path/to/model.gguf", "LLaMA Model File")

    Args:
        asset_path_or_assets: Either a list of asset dicts (each with "name" and "path" keys)
                              or a string path to check (used with asset_name).
        asset_name:           Human-readable label used only in the two-argument form.
    """
    if isinstance(asset_path_or_assets, list):
        for asset in asset_path_or_assets:
            _path = asset.get("path", "")
            _name = asset.get("name", _path)
            if os.path.exists(_path):
                logger.info(f"[FOUND]   {_name}")
            else:
                logger.error(f"[MISSING] {_name} → {_path}")
    else:
        asset_path = asset_path_or_assets
        if os.path.exists(asset_path):
            logger.info(f"[FOUND]   {asset_name}")
        else:
            logger.error(f"[MISSING] {asset_name} → {asset_path}")


def pip_install(*args) -> Tuple[int, str]:
    """
    Run pip install with the current Python interpreter.

    This utility function wraps subprocess.run to install Python packages
    using the same interpreter that's running the notebook. This is safer
    than calling system pip, which might target a different Python installation.

    Args:
        *args: Arguments to pass to pip install (e.g., "torch==2.5.1", "--index-url", "...")

    Returns:
        A tuple (return_code, stderr_output):
            - return_code: 0 on success, non-zero on failure
            - stderr_output: Error messages from pip (empty string if successful)

    Example usage:
        rc, err = pip_install("torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu128")
        if rc == 0:
            print("✅ Installation successful")
        else:
            print(f"❌ Installation failed: {err}")
    """
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet"] + list(args),
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stderr
