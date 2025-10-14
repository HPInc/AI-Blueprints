"""
Utility functions for AI Studio Templates.

This module contains common functions used across notebooks in the project,
including configuration loading, model initialization, and local processing.
"""

import os
import yaml
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from .trt_llm_langchain import TensorRTLangchain


# Default models to be loaded in our examples:
DEFAULT_MODELS = {
    "local": "/home/jovyan/datafabric/meta-llama3.1-8b-Q8/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    "tensorrt": "",
    "hugging-face-local": "meta-llama/Llama-3.2-3B-Instruct",
    "hugging-face-cloud": "mistralai/Mistral-7B-Instruct-v0.3",
}

# Context window sizes for various models
MODEL_CONTEXT_WINDOWS = {
    # LlamaCpp models
    "ggml-model-f16-Q5_K_M.gguf": 4096,
    "ggml-model-7b-q4_0.bin": 4096,
    "gguf-model-7b-4bit.bin": 4096,
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
    "qwen/Qwen-7B": 8192,
    "microsoft/phi-2": 2048,
    "tiiuae/falcon-7b": 4096,
    "meta-llama/Llama-3.2-3B-Instruct": 128000,
    "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf": 4096,
}


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
    from langchain_community.llms import LlamaCpp

    # Fix for Pydantic model rebuild issue
    if hasattr(LlamaCpp, "model_rebuild"):
        LlamaCpp.model_rebuild()

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
            "text-generation",
            model=hf_model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            device=0,
            return_full_text=False,
            do_sample=True,
            temperature=0.1,
        )
        model = HuggingFacePipeline(pipeline=pipe)

    elif model_source == "tensorrt":
        # If a Hugging Face model is specified, it will be used - otherwise, it will try loading the model from local_path
        try:
            import tensorrt_llm

            sampling_params = tensorrt_llm.SamplingParams(
                temperature=0.1, top_p=0.95, max_tokens=512
            )
            if hf_repo_id != "":
                return TensorRTLangchain(
                    model_path=hf_repo_id, sampling_params=sampling_params
                )
            else:
                model_config = os.path.join(local_model_path, "config.json")
                if os.path.isdir(local_model_path) and os.path.isfile(model_config):
                    return TensorRTLangchain(
                        model_path=local_model_path, sampling_params=sampling_params
                    )
                else:
                    raise Exception("Model format incompatible with TensorRT LLM")
        except ImportError:
            raise ImportError(
                "Could not import tensorrt-llm library. "
                "Please make sure tensorrt-llm is installed properly, or "
                "consider using workspaces based on the NeMo Framework"
            )
    elif model_source == "local":
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        # For LlamaCpp, get the context window from the filename
        model_filename = os.path.basename(local_model_path)
        if model_filename in MODEL_CONTEXT_WINDOWS:
            context_window = MODEL_CONTEXT_WINDOWS[model_filename]
        else:
            # Default context window for LlamaCpp models (explicitly set)
            context_window = 4096

        model = LlamaCpp(
            model_path=local_model_path,
            n_gpu_layers=-1,
            n_batch=512,
            n_ctx=context_window,
            max_tokens=1024,
            f16_kv=True,
            callback_manager=callback_manager,
            verbose=False,
            stop=[],
            streaming=False,
            temperature=0,
            use_mmap=False,
        )
    else:
        raise ValueError(f"Unsupported model source: {model_source}")

    # Store context window as model attribute for easy access
    if model and hasattr(model, "__dict__"):
        model.__dict__["_context_window"] = context_window

    return model


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


def dynamic_retriever(
    query: str, collection, top_n: int = None, context_window: int = None
) -> List:
    """
    Retrieve relevant documents with dynamic adaptation based on context window.

    This function automatically determines how many documents to retrieve based on
    the available context window, optimizing for the specific model being used.

    Args:
        query: The search query
        collection: Vector database collection to search in
        top_n: Number of documents to retrieve (if None, will be determined dynamically)
        context_window: Size of the model's context window in tokens

    Returns:
        List: Document objects containing relevant content
    """
    from langchain.schema import Document

    # Dynamically determine how many documents to retrieve based on context window
    if top_n is None:
        if context_window:
            # Larger context windows can handle more documents
            # Using a heuristic: 1 document per 1000 tokens of context
            # with a minimum of 2 and maximum of 10
            suggested_top_n = max(2, min(10, context_window // 1000))
            top_n = suggested_top_n
        else:
            # Default if we can't determine context window
            top_n = 3

    # Check if collection is a Chroma vector store
    if hasattr(collection, "as_retriever"):
        # It's a LangChain Chroma vector store
        retriever = collection.as_retriever(search_kwargs={"k": top_n})
        documents = retriever.get_relevant_documents(query)
    elif hasattr(collection, "_collection"):
        # It's a direct ChromaDB collection
        results = collection._collection.query(query_texts=[query], n_results=top_n)

        # Convert to Document objects
        documents = [
            Document(
                page_content=str(results["documents"][0][i]),
                metadata=(
                    results["metadatas"][0][i]
                    if isinstance(results["metadatas"][0][i], dict)
                    else results["metadatas"][0][i]
                ),
            )
            for i in range(len(results["documents"][0]))
        ]
    else:
        # Try direct query as a fallback
        try:
            results = collection.query(query_texts=[query], n_results=top_n)

            # Convert to Document objects
            documents = [
                Document(
                    page_content=str(results["documents"][i]),
                    metadata=(
                        results["metadatas"][i]
                        if isinstance(results["metadatas"][i], dict)
                        else results["metadatas"][i][0]
                    ),
                )
                for i in range(len(results["documents"]))
            ]
        except AttributeError:
            # If all else fails, raise a more helpful error
            raise AttributeError(
                "The collection object doesn't have required retrieval methods. "
                "Expected a LangChain Chroma vector store or a ChromaDB collection."
            )

    return documents


def format_docs_with_adaptive_context(docs, context_window: int = None) -> str:
    """
    Format retrieved documents using dynamic allocation based on model context window.

    This function:
    1. Adapts to the model's context window size
    2. Keeps full content for the most relevant document when possible
    3. Distributes remaining context based on document relevance
    4. Preserves code structure by breaking at logical points
    5. Provides diagnostics about context usage

    Args:
        docs: List of Document objects to format
        context_window: Size of the model's context window in tokens (if provided)

    Returns:
        Formatted context string for the LLM
    """
    if not docs:
        return ""

    # Average characters per token (this is an approximation)
    chars_per_token = 4

    # Determine the maximum character budget based on context window
    if context_window:
        # Reserve 20% for the prompt and response
        available_tokens = int(context_window * 0.8)
        max_total_chars = available_tokens * chars_per_token
    else:
        # Default conservative estimate if we don't know the context window
        max_total_chars = 8000

    # Track metrics for diagnostic output
    formatted_docs = []
    total_chars = 0
    doc_allocation = []

    # Process documents by relevance order
    for i, doc in enumerate(docs):
        content = doc.page_content
        original_length = len(content)

        # Distribute context budget based on relevance
        # First document gets up to 50% of remaining budget, but don't exceed its actual size
        if i == 0:
            # Give the first (most relevant) document up to 50% of the budget
            budget_fraction = 0.5
        else:
            # Distribute remaining budget exponentially declining by relevance
            budget_fraction = 0.5 / (2**i)

        chars_to_allocate = min(
            int(max_total_chars * budget_fraction),  # Relevance-based allocation
            original_length,  # Don't allocate more than needed
            max_total_chars - total_chars,  # Don't exceed remaining budget
        )

        # If we can fit the whole document, do it
        if original_length <= chars_to_allocate:
            formatted_docs.append(content)
            used_chars = original_length
            truncated = False
        # Otherwise, truncate it
        elif chars_to_allocate > 0:
            # Try to break at a logical point like a line break
            truncation_point = min(chars_to_allocate, original_length)

            # Find a good break point - prefer newlines, then periods, then spaces
            last_newline = content[:truncation_point].rfind("\n")
            last_period = content[:truncation_point].rfind(".")
            last_space = content[:truncation_point].rfind(" ")

            # Use the best break point that's not too far from target (at least 80% of target)
            threshold = truncation_point * 0.8
            if last_newline > threshold:
                truncation_point = last_newline + 1  # +1 to include the newline
            elif last_period > threshold:
                truncation_point = last_period + 1  # +1 to include the period
            elif last_space > threshold:
                truncation_point = last_space + 1  # +1 to include the space

            formatted_content = f"{content[:truncation_point]}... (truncated)"
            formatted_docs.append(formatted_content)
            used_chars = truncation_point + 15  # +15 for the truncation message
            truncated = True
        else:
            # No budget left for this document
            break

        # Track allocation for diagnostic output
        doc_allocation.append(
            {
                "document": i + 1,
                "original_chars": original_length,
                "allocated_chars": used_chars,
                "truncated": truncated,
                "percent_used": (
                    round(100 * used_chars / original_length, 1)
                    if original_length > 0
                    else 100
                ),
            }
        )

        total_chars += used_chars

        # Stop if we've reached our budget
        if total_chars >= max_total_chars:
            break

    # Join everything together with clear separators
    formatted_text = "\n\n".join(formatted_docs)

    return formatted_text
