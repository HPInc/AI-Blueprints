# ─────── Standard Library Imports ───────
import base64  # Encoding and decoding binary data
import logging  # Logging utilities
import sys  # System-specific parameters and functions
import time  # Time-related utilities
from functools import wraps  # Function decorators support
import os
import yaml
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

# ─────── Third-Party Package Imports ───────
from IPython.display import (
    HTML,
    display,
)  # Rich HTML display utilities for Jupyter environments

# Default models to be loaded in our examples:
DEFAULT_MODELS = {
    "local": "/home/jovyan/datafabric/meta-llama3.1-8b-Q8/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    "qwen-local": "/home/jovyan/datafabric/Qwen2.5-Omni-7B/",
    "clap-local": "/home/jovyan/datafabric/clap-htsat-unfused/",
    "hugging-face-cloud": ["Qwen/Qwen2.5-Omni-7B", "laion/clap-htsat-unfused"],
}

# Context window sizes for various models
MODEL_CONTEXT_WINDOWS = {
    # HuggingFace models
    "mistralai/Mistral-7B-Instruct-v0.3": 8192,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 4096,
    "meta-llama/Llama-2-7b-chat-hf": 4096,
    "meta-llama/Llama-3-8b-chat-hf": 8192,
    "google/flan-t5-base": 512,
    "google/flan-t5-large": 512,
    "TheBloke/WizardCoder-Python-7B-V1.0-GGUF": 4096,
    # OpenAI models
    "gpt-3.5-turbo": 16385,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    # Anthropic models
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 180000,
    "claude-3-haiku-20240307": 48000,
    # Other models
    "Qwen/Qwen2.5-Omni-7B": 8192,
    "laion/clap-htsat-unfused": 8192,
    "qwen/Qwen-7B": 8192,
    "microsoft/phi-2": 2048,
    "tiiuae/falcon-7b": 4096,
    "meta-llama/Llama-3.2-3B-Instruct": 128000,
    "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf": 4096,
}

# ─────── Color and Emoji mapping per level ───────
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


# ─────── Logger Setup ───────
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


# ─────── Configurations Setup Functions───────
def configure_hf_cache(cache_dir: str = "/home/jovyan/local/hugging_face") -> None:
    """
    Configure HuggingFace cache directories to persist models locally.

    Args:
        cache_dir: Base directory for HuggingFace cache. Defaults to "/home/jovyan/local/hugging_face".
    """
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")


def load_secrets(
    secret_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load secrets from secrets environment variables.
    Args:
        secret_keys: List of expected secret names.
        If None, every project environment variable with 'AIS' prefix is returned.

    Returns:
        Dictionary containing all secrets for the project.

    ValueError:
        Requested secret(s) are missing or none found with AIS- prefix.
    """
    # Build secrets from environment
    if secret_keys is None:
        secrets = {
            k: v for k, v in os.environ.items() if k.isupper() and k.startswith("AIS_")
        }
        if not secrets:
            raise ValueError(
                "No environment variables found with prefix 'AIS_'. "
                "Please set your required project secrets in AIS Secrets Manager."
            )
    else:
        secrets = {k: os.environ.get(k) for k in secret_keys}
        missing = [k for k, v in secrets.items() if v is None]
        if missing:
            raise ValueError(
                f"Provided secrets are missing as environment variables for this project: {', '.join(missing)}"
            )
    return secrets


def load_secrets_to_env(secrets_path: str = "../configs/secrets.yaml") -> None:
    """
    Loads secrets from a YAML file and sets them as environment variables.

    Parameters:
    - secrets_path (str): Path to the secrets YAML file.
    """
    secrets_file = Path(secrets_path).resolve()

    if not secrets_file.exists():
        raise FileNotFoundError(f"Secrets file not found: {secrets_file}")

    with secrets_file.open("r", encoding="utf-8") as file:
        try:
            secrets = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML: {e}")

    if not isinstance(secrets, dict):
        raise ValueError("Secrets file must contain a top-level dictionary.")

    for key, value in secrets.items():
        if not isinstance(key, str):
            raise TypeError(
                f"Environment variable key must be a string. Got: {type(key)}"
            )
        # We are adding "AIS_" prefix for compatibility with HP AI Studio Secrets Manager.
        env_key = key if key.upper().startswith("AIS_") else f"AIS_{key.upper()}"
        os.environ[env_key] = str(value)

    print(f"✅ Loaded {len(secrets)} secrets into environment variables.")


def load_config(config_path: str = "../../configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration YAML file.
        secrets_path: Path to the secrets YAML file.

    Returns:
        Dictionary containing the project configurations.

    Raises:
        FileNotFoundError: If the config file is not found.
    """
    # Convert to absolute paths if needed
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


# ─────── LLM-Related Functions ───────
def initialize_llm(
    model_source: str = "local",
    secrets: Optional[Dict[str, Any]] = None,
    local_model_path: str = DEFAULT_MODELS["local"],
    hf_repo_id: str = "",
) -> Any:
    """
    Initialize a language model based on specified source.

    Args:
        model_source: Source of the model. Options are "local", "hugging-face-local", or "hugging-face-cloud".
        secrets: Dictionary containing API keys for cloud services.
        local_model_path: Path to local model file.

    Returns:
        Initialized language model object.

    Raises:
        ImportError: If required libraries are not installed.
        ValueError: If an unsupported model_source is provided.
    """
    # Check dependencies
    missing_deps = []
    for module in [
        "langchain_huggingface",
        "langchain_core.callbacks",
        "langchain_community.llms",
    ]:
        if not importlib.util.find_spec(module):
            missing_deps.append(module)

    if missing_deps:
        raise ImportError(f"Missing required dependencies: {', '.join(missing_deps)}")

    # Import required libraries
    from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
    from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

    model = None
    context_window = None

    # Initialize based on model source
    if model_source == "hugging-face-cloud":
        if hf_repo_id == "":
            repo_id = DEFAULT_MODELS["hugging-face-cloud"]
        else:
            repo_id = hf_repo_id
        if not secrets or "AIS_HUGGINGFACE_API_KEY" not in secrets:
            raise ValueError("HuggingFace API key is required for cloud model access")

        huggingfacehub_api_token = secrets["AIS_HUGGINGFACE_API_KEY"]
        # Get context window from our lookup table
        if repo_id in MODEL_CONTEXT_WINDOWS:
            context_window = MODEL_CONTEXT_WINDOWS[repo_id]

        model = HuggingFaceEndpoint(
            huggingfacehub_api_token=huggingfacehub_api_token,
            repo_id=repo_id,
            task="audio-rag",
        )

    elif model_source == "hugging-face-local":
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        if "AIS_HUGGINGFACE_API_KEY" in secrets:
            os.environ["HF_TOKEN"] = secrets["AIS_HUGGINGFACE_API_KEY"]
        if hf_repo_id == "":
            model_id = DEFAULT_MODELS["hugging-face-local"]
        else:
            model_id = hf_repo_id
        # Get context window from our lookup table
        if model_id in MODEL_CONTEXT_WINDOWS:
            context_window = MODEL_CONTEXT_WINDOWS[model_id]

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        hf_model = AutoModelForCausalLM.from_pretrained(model_id)

        # If tokenizer has model_max_length, that's our context window
        if hasattr(
            tokenizer, "model_max_length"
        ) and tokenizer.model_max_length not in (None, -1):
            context_window = tokenizer.model_max_length

        # Disable automatic chat template application by removing it from tokenizer
        if hasattr(tokenizer, "chat_template"):
            tokenizer.chat_template = None

        pipe = pipeline(
            "audio-rag",
            model=hf_model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            device=0,
            return_full_text=False,
            do_sample=True,
            temperature=0.1,
        )
        model = HuggingFacePipeline(pipeline=pipe)

    # elif model_source == "local":
    #     callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    #     # For LlamaCpp, get the context window from the filename
    #     model_filename = os.path.basename(local_model_path)
    #     if model_filename in MODEL_CONTEXT_WINDOWS:
    #         context_window = MODEL_CONTEXT_WINDOWS[model_filename]
    #     else:
    #         # Default context window for LlamaCpp models (explicitly set)
    #         context_window = 4096

    #     model = LlamaCpp(
    #         model_path=local_model_path,
    #         n_gpu_layers=-1,
    #         n_batch=512,
    #         n_ctx=context_window,
    #         max_tokens=1024,
    #         f16_kv=True,
    #         callback_manager=callback_manager,
    #         verbose=False,
    #         stop=[],
    #         streaming=False,
    #         temperature=0.2,
    #         use_mmap=False,
    #     )
    else:
        raise ValueError(f"Unsupported model source: {model_source}")

    # Store context window as model attribute for easy access
    if model and hasattr(model, "__dict__"):
        model.__dict__["_context_window"] = context_window

    return model


def get_response_from_llm(llm, system_prompt, user_prompt):
    model_prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    return llm(model_prompt)


# ─────── Helper Functions ───────
# def ensure_wav(input_path: str) -> str:
#     p = Path(input_path)
#     if p.suffix.lower() == ".wav":
#         return str(p)
#     out = str(p.with_suffix(".wav"))
#     try:
#         subprocess.run(
#             ["ffmpeg", "-y", "-i", str(p), "-ar", "16000", "-ac", "1", out],
#             check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
#         )
#         return out if Path(out).exists() else str(p)
#     except Exception:
#         return str(p)


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def get_models_dir():
    """Get or create the models directory for downloaded models"""
    models_dir = get_project_root() / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def format_model_path(model_id: str) -> Path:
    """Convert a HuggingFace model ID to a local path"""
    return get_models_dir() / model_id.replace("/", "__")


def get_model_cache_dir():
    """Get the directory for caching downloaded models"""
    cache_dir = get_models_dir() / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def setup_model_environment():
    """Setup model-related environment variables for the project"""
    # Configure HuggingFace cache to use project directory
    hf_cache_dir = str(get_model_cache_dir())
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["HF_HUB_CACHE"] = hf_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir

    # Optimize HuggingFace behavior for deployment
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # Reduce console noise
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"  # Disable telemetry

    # Configure tokenizers cache
    tokenizers_cache = str(get_model_cache_dir() / "tokenizers")
    os.environ["TOKENIZERS_CACHE"] = tokenizers_cache

    logger.info(f"✅ Model environment configured with cache: {hf_cache_dir}")


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


def login_huggingface(secrets: Dict[str, Any]) -> None:
    """
    Login to Hugging Face using token from secrets.

    Args:
        secrets: Dictionary containing the Hugging Face token.

    Raises:
        ValueError: If the token is missing.
    """
    from huggingface_hub import login

    token = secrets.get("AIS_HUGGINGFACE_API_KEY")
    if not token:
        raise ValueError("❌ Hugging Face token not found in secrets.yaml.")

    login(token=token)
    print("✅ Logged into Hugging Face successfully.")


def clean_code(result: str) -> str:
    """
    Clean code extraction function that handles various formats.

    Args:
        result: The raw text output from an LLM that may contain code.

    Returns:
        str: Cleaned code without markdown formatting or explanatory text.
    """
    if not result or not isinstance(result, str):
        return ""

    # Remove common prefixes and wrapper text
    prefixes = [
        "Answer:",
        "Expected Answer:",
        "Expected Output:",
        "Python code:",
        "Here's the code:",
        "My Response:",
        "Response:",
    ]
    for prefix in prefixes:
        if result.lstrip().startswith(prefix):
            result = result.replace(prefix, "", 1)

    # Handle markdown code blocks
    if "```python" in result or "```" in result:
        # Extract code between markdown code blocks
        code_blocks = []
        in_code_block = False
        lines = result.split("\n")
        current_block = []

        for line in lines:
            if line.strip().startswith("```"):
                if in_code_block:
                    # End of block, add it to our list
                    code_blocks.append("\n".join(current_block))
                    current_block = []
                in_code_block = not in_code_block
                continue

            if in_code_block:
                current_block.append(line)

        if code_blocks:
            # Use the longest code block found
            result = max(code_blocks, key=len)
        else:
            # Fallback to simple replacement if block extraction fails
            result = result.replace("```python", "").replace("```", "")

    # Remove any remaining explanatory text before or after the code
    lines = result.split("\n")
    code_lines = []
    in_code_block = False

    # First, look for the first actual code line
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and (
            stripped.startswith("import ")
            or stripped.startswith("from ")
            or stripped.startswith("def ")
            or stripped.startswith("class ")
        ):
            in_code_block = True
            lines = lines[i:]  # Start from this line
            break

    # Now process all the lines
    for line in lines:
        stripped = line.strip()
        # Skip empty lines at the beginning
        if not stripped and not code_lines:
            continue

        # Ignore lines that appear to be LLM "thinking" or explanations
        if any(
            text in stripped.lower()
            for text in ["here's", "i'll", "please provide", "this code will"]
        ):
            if not any(
                code_indicator in stripped
                for code_indicator in ["import ", "def ", "class ", "="]
            ):
                continue

        # If we see code-like content, include it
        if stripped and (
            stripped.startswith("import ")
            or stripped.startswith("from ")
            or stripped.startswith("def ")
            or stripped.startswith("class ")
            or "=" in stripped
            or stripped.startswith("#")
            or "(" in stripped
            or "." in stripped
            and not stripped.endswith(".")
            or stripped.startswith("with ")
            or stripped.startswith("if ")
            or stripped.startswith("for ")
            or stripped.startswith("while ")
            or stripped.startswith("@")
        ):
            in_code_block = True
            code_lines.append(line)
        # Include indented lines or lines continuing code
        elif stripped and (
            in_code_block or line.startswith(" ") or line.startswith("\t")
        ):
            code_lines.append(line)

    cleaned_code = "\n".join(code_lines).strip()

    # One last check - if the cleaned code starts with text that looks like a response,
    # try to find the first actual code statement
    first_lines = cleaned_code.split("\n", 5)
    for i, line in enumerate(first_lines):
        if line.strip().startswith(("import ", "from ", "def ", "class ")):
            if i > 0:
                cleaned_code = "\n".join(first_lines[i:] + cleaned_code.split("\n")[5:])
            break

    return cleaned_code


def generate_code_with_retries(
    chain, example_input, callbacks=None, max_attempts=3, min_code_length=10
):
    """
    Execute a chain with retry logic for empty or short responses.

    Args:
        chain: The LangChain chain to execute.
        example_input: Input dictionary with query and question.
        callbacks: Optional callbacks to pass to the chain.
        max_attempts: Maximum number of attempts before giving up.
        min_code_length: Minimum acceptable code length.

    Returns:
        tuple: (raw_output, clean_code_output)
    """
    import time

    attempts = 0
    output = None

    while attempts < max_attempts:
        attempts += 1
        try:
            # Add a small delay before each attempt (only needed for retries)
            if attempts > 1:
                time.sleep(1)  # Small delay between retries

            # Invoke the chain
            output = chain.invoke(
                example_input, config=dict(callbacks=callbacks) if callbacks else {}
            )

            # Clean the code
            clean_code_output = clean_code(output)

            # Only continue with retry if we got no usable output
            if clean_code_output and len(clean_code_output) > min_code_length:
                break

            print(f"Attempt {attempts}: Output too short or empty, retrying...")

        except Exception as e:
            print(f"Error in attempt {attempts}: {str(e)}")
            if attempts == max_attempts:
                raise

    return output, clean_code_output


def get_model_context_window(model) -> int:
    """
    Get context window using model identifier and lookup table.

    This function simplifies context window resolution by using a lookup table

    1. For LlamaCpp models: extract the filename from model_path and check in MODEL_CONTEXT_WINDOWS
    2. For HuggingFace models: check the repo_id in MODEL_CONTEXT_WINDOWS
    3. Fall back to explicit parameters if available
    4. Try to get context window from a stored attribute (_context_window) on the model
    5. Use a default conservative estimate if all else fails

    Args:
        model: Any language model object (LlamaCpp, HuggingFace, OpenAI, etc.)

    Returns:
        int: The determined context window size in tokens, defaulting to 2048 if detection fails
    """
    # Check if we already stored the context window in the model itself
    if hasattr(model, "_context_window") and model._context_window is not None:
        return model._context_window

    # For LlamaCpp: extract filename from model_path
    if hasattr(model, "model_path"):
        model_filename = os.path.basename(model.model_path)
        if model_filename in MODEL_CONTEXT_WINDOWS:
            return MODEL_CONTEXT_WINDOWS[model_filename]

    # For HuggingFace models: check repo_id
    if hasattr(model, "repo_id"):
        if model.repo_id in MODEL_CONTEXT_WINDOWS:
            return MODEL_CONTEXT_WINDOWS[model.repo_id]

    # Fall back to direct n_ctx attribute if available
    if hasattr(model, "n_ctx"):
        return model.n_ctx

    # Check model_kwargs for context window parameters
    if hasattr(model, "model_kwargs"):
        kwargs = model.model_kwargs
        for param_name in ["n_ctx", "max_tokens", "max_length", "context_window"]:
            if param_name in kwargs and kwargs[param_name] is not None:
                return kwargs[param_name]

    # For HuggingFace pipeline models: check tokenizer
    if (
        hasattr(model, "pipeline")
        and hasattr(model.pipeline, "tokenizer")
        and hasattr(model.pipeline.tokenizer, "model_max_length")
    ):
        if (
            model.pipeline.tokenizer.model_max_length > 0
            and model.pipeline.tokenizer.model_max_length < 1000000000000000
        ):
            return model.pipeline.tokenizer.model_max_length

    # Use a very conservative default if all detection methods fail
    return 2048


def get_context_window(model) -> int:
    """
    Get context window size from model.

    This function first checks for the explicit _context_window attribute
    that we set during initialization, then falls back to the more
    complex detection logic if needed.

    Args:
        model: Any language model object

    Returns:
        int: The context window size in tokens
    """
    if hasattr(model, "_context_window") and model._context_window is not None:
        return model._context_window

    # Fall back to detection logic
    return get_model_context_window(model)


# ───────────────────────── Audio-Specific Helper Functions ──────────────────────────
def sec_to_timestamp(seconds: float) -> str:
    """
    Convert a float number of seconds to HH:MM:SS.mmm for UI highlighting.
    """
    ms = int((seconds - int(seconds)) * 1000)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def slice_with_timestamps(
    text: str, start_idx: int, end_idx: int, ratio: float = 0.01
) -> dict:
    """
    Create a small excerpt dict that the UI can render as evidence.

    Args:
        text      : full transcript chunk.
        start_idx : absolute start index (char) inside original transcript.
        end_idx   : absolute end index   (char) inside original transcript.
        ratio     : seconds per character – Whisper returns timestamps
                    in seconds while we work with character offsets.

    Returns:
        {"start": "00:01:23.456", "end": "00:01:27.890", "text": "...excerpt..."}
    """
    start_ts = sec_to_timestamp(start_idx * ratio)
    end_ts = sec_to_timestamp(end_idx * ratio)
    preview = text[:180].replace("\n", " ") + ("…" if len(text) > 180 else "")
    return {"start": start_ts, "end": end_ts, "text": preview}


def initialize_audio_models(
    config: Dict[str, Any], secrets: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Any, Any, Any]:
    """
    Initialize both Qwen and CLAP models based on configuration.

    This function follows the standard AI Studio pattern used by other blueprints:
    1. Try to load from local datafabric paths first
    2. Fallback to remote download if local models not found
    3. Use ModelSelector for consistent model management

    Args:
        config: Configuration dictionary containing model paths and settings
        secrets: Optional secrets for HuggingFace authentication

    Returns:
        Tuple of (qwen_model, qwen_processor, clap_model, clap_processor)
    """
    from .model_selection import ModelSelector

    model_source = config.get("model_source", "local")

    qwen_model, qwen_processor = None, None
    clap_model, clap_processor = None, None

    if model_source == "local":
        qwen_path = config.get("qwen_model_path", DEFAULT_MODELS["qwen-local"])
        clap_path = config.get("clap_model_path", DEFAULT_MODELS["clap-local"])

        qwen_exists = Path(qwen_path).exists()
        clap_exists = Path(clap_path).exists()

        logger.info(f"Checking local models: Qwen={qwen_exists}, CLAP={clap_exists}")

        if qwen_exists and clap_exists:
            logger.info("✅ Loading models from local datafabric")
            try:
                # Load Qwen from local path
                from transformers import (
                    Qwen2_5OmniProcessor,
                    Qwen2_5OmniThinkerForConditionalGeneration,
                )

                qwen_processor = Qwen2_5OmniProcessor.from_pretrained(qwen_path)
                qwen_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                    qwen_path, torch_dtype="auto", device_map="auto"
                )

                # Load CLAP from local path
                from transformers import AutoProcessor, ClapModel

                clap_processor = AutoProcessor.from_pretrained(clap_path)
                clap_model = ClapModel.from_pretrained(clap_path)

                logger.info("✅ Successfully loaded models from local datafabric")
                return qwen_model, qwen_processor, clap_model, clap_processor

            except Exception as e:
                logger.warning(f"⚠️ Failed to load local models: {e}")

        # If local loading failed or models don't exist, try remote
        if not (qwen_exists and clap_exists) or (qwen_model is None):
            logger.info("📡 Falling back to remote model download")
            model_source = "hugging-face-cloud"

    # Load from remote (either direct request or fallback)
    if model_source in ["hugging-face-cloud", "hugging-face-local"]:
        logger.info("📡 Loading models from HuggingFace")

        # Setup HuggingFace authentication if available
        if secrets and "AIS_HUGGINGFACE_API_KEY" in secrets:
            os.environ["AIS_HUGGINGFACE_API_KEY"] = secrets["AIS_HUGGINGFACE_API_KEY"]

        selector = ModelSelector()

        # Load Qwen
        logger.info("Loading Qwen/Qwen2.5-Omni-7B...")
        selector.select_model("Qwen/Qwen2.5-Omni-7B")
        qwen_model = selector.get_model()
        qwen_processor = selector.get_processor()

        # Load CLAP
        logger.info("Loading laion/clap-htsat-unfused...")
        selector.select_model("laion/clap-htsat-unfused")
        clap_model = selector.get_model()
        clap_processor = selector.get_processor()

        logger.info("✅ Successfully loaded models from HuggingFace")

    return qwen_model, qwen_processor, clap_model, clap_processor
