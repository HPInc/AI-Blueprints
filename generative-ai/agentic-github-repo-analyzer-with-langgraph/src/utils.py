# ─────── Standard Library Imports ───────
import base64  # Encoding and decoding binary data
import logging  # Logging utilities
import os  # Operating system interaction
import sys  # System-specific parameters and functions
import time  # Time-related utilities
from functools import wraps  # Function decorators support

# ─────── Third-Party Package Imports ───────
import yaml  # YAML file parsing
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


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration data
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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
        result = (
            loaded.predict(input_df, params=params)
            if params
            else loaded.predict(input_df)
        )

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
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(
            f"✅ Done — VRAM: {used:.1f} GB / {total:.1f} GB ({used / total * 100:.0f}% used)"
        )
    else:
        print("✅ Done — CUDA not available, CPU memory released via GC")
