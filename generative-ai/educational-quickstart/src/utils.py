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

    Why ChatPromptTemplate instead of a raw f-string?
        Raw f-strings embed user content directly into special tokens, which can
        produce malformed prompts when the document text contains special characters.
        ChatPromptTemplate validates and escapes variables before the model sees them,
        following the same LCEL chain pattern used across all other blueprints.

    Why StrOutputParser?
        In LangChain 1.x, llm.invoke() may return a generation metadata object.
        StrOutputParser guarantees a plain Python str, preventing downstream type errors.

    Args:
        llm: A LlamaCpp model instance (loaded in the notebook)
        system_prompt: The instruction telling the AI how to behave
                       (e.g., "You are a helpful assistant")
        user_prompt: The actual question or input from the user

    Returns:
        The model's text response as a string

    Learn more about LCEL chains:
        https://python.langchain.com/docs/concepts/lcel/
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # Meta-Llama 3.1 chat template — same format used in vanilla-rag-with-langchain
    meta_llama_template = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    prompt = ChatPromptTemplate.from_template(meta_llama_template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"system_prompt": system_prompt, "user_prompt": user_prompt})


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


def _isolated_predict_worker(
    model_uri: str,
    input_df,
    params,
    src_path: str,
    result_path: str,
    error_queue,
) -> None:
    """
    Worker function that runs inside a spawned subprocess.

    Must be a module-level function (not a lambda or nested def) so that Python's
    multiprocessing 'spawn' context can pickle and send it to the child process.

    Why write to a temp file instead of using a Queue?
        Large results (e.g. base64-encoded images, several MB) overflow the OS
        pipe buffer that backs a multiprocessing.Queue. When that happens the
        child blocks on queue.put() while the parent blocks on p.join() —
        a classic deadlock. Writing to a temp file bypasses the pipe entirely.
    """
    try:
        import sys
        sys.path.insert(0, src_path)

        import mlflow
        import pickle

        loaded = mlflow.pyfunc.load_model(model_uri=model_uri)
        result = loaded.predict(input_df, params=params) if params else loaded.predict(input_df)

        with open(result_path, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        import traceback
        error_queue.put(traceback.format_exc())


def run_isolated_mlflow_predict(
    model_uri: str,
    input_df,
    params=None,
    src_path: str = "..",
    timeout: int = 600,
):
    """
    Load an MLflow model and run ``predict()`` inside an isolated subprocess.

    Why subprocess instead of loading directly?
        When ``mlflow.pyfunc.load_model()`` is called in the same notebook process,
        the model's CUDA tensors remain in VRAM until Python's garbage collector
        decides to free them — which is unpredictable. With a subprocess, the CUDA
        driver reclaims **all** VRAM the instant the child process exits, with zero
        manual cleanup required.

    Why ``spawn`` and not ``fork``?
        CUDA explicitly forbids ``fork`` after device initialization — it corrupts
        the GPU context in the child. ``spawn`` starts a completely fresh Python
        interpreter with no inherited CUDA state, which is safe.

    Why a temp file for the result instead of a Queue?
        Large payloads (e.g. base64-encoded images) overflow the OS pipe buffer
        that backs a multiprocessing.Queue, causing a deadlock where the child
        blocks on put() and the parent blocks on join(). A temp file has no size
        limit and avoids the issue entirely.

    Args:
        model_uri:  MLflow model URI (e.g. ``"runs:/<run_id>/artifact"``).
        input_df:   ``pandas.DataFrame`` to pass to ``predict()``.
        params:     Optional params dict forwarded to ``predict(params=...)``.
        src_path:   Path inserted into ``sys.path`` so ``src.*`` imports work
                    inside the subprocess (default: ``".."``).
        timeout:    Seconds to wait before killing the subprocess (default 600).

    Returns:
        The ``pandas.DataFrame`` returned by the model's ``predict()`` method.

    Raises:
        RuntimeError:  If the subprocess raises an exception.
        TimeoutError:  If the subprocess exceeds ``timeout`` seconds.

    Example::

        result = run_isolated_mlflow_predict(
            model_uri=model_uri,
            input_df=pd.DataFrame([{"question": "What is AI?"}]),
        )
        print(result["answer"].iloc[0])
    """
    import multiprocessing as mp
    import tempfile
    import pickle
    import os

    # Temp file to receive the DataFrame result from the subprocess
    fd, result_path = tempfile.mkstemp(suffix=".pkl", prefix="mlflow_result_")
    os.close(fd)

    # 'spawn' starts a fresh interpreter — required for CUDA safety
    ctx = mp.get_context("spawn")
    error_queue = ctx.Queue()

    p = ctx.Process(
        target=_isolated_predict_worker,
        args=(model_uri, input_df, params, src_path, result_path, error_queue),
    )

    print("🚀 Starting isolated subprocess for MLflow inference...")
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        os.unlink(result_path)
        raise TimeoutError(
            f"Isolated predict subprocess timed out after {timeout}s.\n"
            "Try increasing the `timeout` argument."
        )

    if not error_queue.empty():
        os.unlink(result_path)
        raise RuntimeError(
            f"Error inside isolated predict subprocess:\n{error_queue.get()}"
        )

    try:
        with open(result_path, "rb") as f:
            result = pickle.load(f)
        return result
    except Exception as e:
        raise RuntimeError(
            f"Failed to read result from subprocess temp file: {e}\n"
            f"Exit code: {p.exitcode}"
        )
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass


def _is_heavy(obj) -> bool:
    """
    Return True if ``obj`` is a heavyweight GPU/C++ object worth releasing.

    Covers:
    - ``torch.nn.Module`` — any PyTorch layer, encoder, transformer, VAE, etc.
    - Objects whose type name contains "llama" — llama_cpp C++ handles.
    - Objects whose type name contains "pipeline" — diffusers Pipeline objects.
    """
    try:
        import torch
        if isinstance(obj, torch.nn.Module):
            return True
    except ImportError:
        pass
    type_name = type(obj).__name__.lower()
    return any(kw in type_name for kw in ("llama", "pipeline", "diffusion"))


def _release_inner(m) -> None:
    """
    Null out and delete every heavy sub-component found on a model object.

    Instead of a hardcoded list of attribute names, this function inspects all
    attributes dynamically and releases anything that looks like a GPU/C++ object
    (``torch.nn.Module``, llama_cpp handles, diffusers pipelines).
    Works on both direct model instances and nested objects like ``._pipeline``.
    """
    for attr in list(vars(m).keys()):
        try:
            obj = getattr(m, attr, None)
        except Exception:
            continue
        if obj is None:
            continue
        # Recurse one level into known container attributes (e.g. ._pipeline)
        if hasattr(obj, "__dict__") and not _is_heavy(obj):
            _release_inner(obj)
        if _is_heavy(obj):
            try:
                setattr(m, attr, None)
                del obj
            except Exception:
                pass


def release_model_vram(*models, label: str = "model") -> None:
    """
    Release GPU VRAM occupied by one or more model objects.

    Use this between notebook sections to avoid CUDA Out-of-Memory errors when
    loading a second model (e.g., after demo inference, before loading the
    registered model from MLflow).

    How it works:
        1. For each model, nullifies heavy sub-components stored in ``_pipeline``
           (text encoders, transformer, VAE) and the underlying C++ LLM client
           so Python's garbage collector can reclaim the CUDA tensors immediately.
        2. Unwraps MLflow ``PyFuncModel`` wrappers automatically — works the same
           whether you pass a direct model or a ``mlflow.pyfunc.load_model()`` result.
        3. Runs a multi-pass garbage collection and flushes the CUDA memory cache.

    .. important::

        After calling this function you **must** also ``del`` the variable in the
        notebook cell::

            release_model_vram(loaded_model, label="registered model")
            del loaded_model   # ← required!

        ``release_model_vram`` nulls the internal tensors, but only ``del`` in
        the calling scope removes the last Python reference so the GC can
        actually return the VRAM to the OS/driver.

    Args:
        *models: One or more model objects to release. Accepts direct model
                 instances or ``mlflow.pyfunc.PyFuncModel`` wrappers.
        label:   Human-readable name shown in the progress output (default "model").

    Example::

        # Before loading the registered model — avoids double VRAM usage:
        release_model_vram(model, label="demo model")
        del model

        # After verification — free the registered model too:
        release_model_vram(loaded_model, label="registered model")
        del loaded_model
    """
    import gc

    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    print(f"🧹 Releasing VRAM for: {label} ...")

    for m in models:
        _release_inner(m)
        del m

    # ── Multi-pass GC + CUDA flush ─────────────────────────────────────────
    for _ in range(3):
        gc.collect()

    if has_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        used  = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ Done — VRAM: {used:.1f} GB / {total:.1f} GB ({used / total * 100:.0f}% used)")
    else:
        print("✅ Done — CUDA not available, CPU memory released via GC")


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
