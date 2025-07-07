"""
Utility functions for ORPO fine-tuning templates.

This module contains common functions used across notebooks in the project,
including configuration loading, model initialization, and helper functions.
"""

import os
import yaml
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

try:
    # First try absolute import
    from trt_llm_langchain import TensorRTLangchain
except ImportError:
    # If that fails, try adding the src directory to path and import
    import sys
    src_dir = str(Path(__file__).parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from trt_llm_langchain import TensorRTLangchain


#Default models to be loaded in our examples:
DEFAULT_MODELS = {
    "local": "/home/jovyan/datafabric/meta-llama3.1-8b-Q8/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    "tensorrt": "",
    "hugging-face-local": "meta-llama/Llama-3.2-3B-Instruct",
    "hugging-face-cloud": "mistralai/Mistral-7B-Instruct-v0.3"
}

# Context window sizes for various models
MODEL_CONTEXT_WINDOWS = {
    # LlamaCpp models
    'ggml-model-f16-Q5_K_M.gguf': 4096,
    'ggml-model-7b-q4_0.bin': 4096,
    'gguf-model-7b-4bit.bin': 4096,

    # HuggingFace models
    'mistralai/Mistral-7B-Instruct-v0.3': 8192,
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B': 4096,
    'meta-llama/Llama-2-7b-chat-hf': 4096,
    'meta-llama/Llama-3-8b-chat-hf': 8192,
    'google/flan-t5-base': 512,
    'google/flan-t5-large': 512,
    'TheBloke/WizardCoder-Python-7B-V1.0-GGUF': 4096,

    # OpenAI models
    'gpt-3.5-turbo': 16385,
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'gpt-4-turbo': 128000,
    'gpt-4o': 128000,

    # Anthropic models
    'claude-3-opus-20240229': 200000,
    'claude-3-sonnet-20240229': 180000,
    'claude-3-haiku-20240307': 48000,

    # Other models
    'qwen/Qwen-7B': 8192,
    'microsoft/phi-2': 2048,
    'tiiuae/falcon-7b': 4096,
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


def load_config_and_secrets(
    config_path: str = "../../config/config.yaml",
    secrets_path: str = "../../config/secrets.yaml"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load configuration and secrets from YAML files.

    Args:
        config_path: Path to the configuration YAML file.
        secrets_path: Path to the secrets YAML file.

    Returns:
        Tuple containing (config, secrets) as dictionaries.

    Raises:
        FileNotFoundError: If either the config or secrets file is not found.
    """
    # Convert to absolute paths if needed
    config_path = os.path.abspath(config_path)
    secrets_path = os.path.abspath(secrets_path)

    if not os.path.exists(secrets_path):
        raise FileNotFoundError(f"secrets.yaml file not found in path: {secrets_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml file not found in path: {config_path}")

    with open(config_path) as file:
        config = yaml.safe_load(file)

    with open(secrets_path) as file:
        secrets = yaml.safe_load(file)

    return config, secrets


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
    hf_repo_id: str = ""
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
    for module in ["langchain_huggingface", "langchain_core.callbacks", "langchain_community.llms"]:
        if not importlib.util.find_spec(module):
            missing_deps.append(module)
    
    if missing_deps:
        raise ImportError(f"Missing required dependencies: {', '.join(missing_deps)}")
    
    # Import required libraries
    from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
    from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
    from langchain_community.llms import LlamaCpp

    model = None
    context_window = None
    
    # Initialize based on model source
    if model_source == "hugging-face-cloud":
        if hf_repo_id == "":
            repo_id = DEFAULT_MODELS["hugging-face-cloud"]
        else:
            repo_id = hf_repo_id  
        if not secrets or "HUGGINGFACE_API_KEY" not in secrets:
            raise ValueError("HuggingFace API key is required for cloud model access")
            
        huggingfacehub_api_token = secrets["HUGGINGFACE_API_KEY"]
        # Get context window from our lookup table
        if repo_id in MODEL_CONTEXT_WINDOWS:
            context_window = MODEL_CONTEXT_WINDOWS[repo_id]

        model = HuggingFaceEndpoint(
            huggingfacehub_api_token=huggingfacehub_api_token,
            repo_id=repo_id,
            task="text-generation",
        )

    elif model_source == "hugging-face-local":
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        if "HUGGINGFACE_API_KEY" in secrets:
            os.environ["HF_TOKEN"] = secrets["HUGGINGFACE_API_KEY"]
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
        if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length not in (None, -1):
            context_window = tokenizer.model_max_length

        # Disable automatic chat template application by removing it from tokenizer
        if hasattr(tokenizer, 'chat_template'):
            tokenizer.chat_template = None

        pipe = pipeline(
            "text-generation", 
            model=hf_model, 
            tokenizer=tokenizer, 
            max_new_tokens=100, 
            device=0,
            return_full_text=False,
            do_sample=True,
            temperature=0.1
        )
        # Create HuggingFacePipeline without automatic chat template application
        model = HuggingFacePipeline(pipeline=pipe)
        
    elif model_source == "tensorrt":
        #If a Hugging Face model is specified, it will be used - otherwise, it will try loading the model from local_path
        try:
            import tensorrt_llm
            sampling_params = tensorrt_llm.SamplingParams(temperature=0.1, top_p=0.95, max_tokens=512) 
            if hf_repo_id != "":
                return TensorRTLangchain(model_path = hf_repo_id, sampling_params = sampling_params)
            else:
                model_config = os.path.join(local_model_path, "config.json")
                if os.path.isdir(local_model_path) and os.path.isfile(model_config):
                    return TensorRTLangchain(model_path = local_model_path, sampling_params = sampling_params)
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
            temperature=0.2,
        )
    else:
        raise ValueError(f"Unsupported model source: {model_source}")

    # Store context window as model attribute for easy access
    if model and hasattr(model, '__dict__'):
        model.__dict__['_context_window'] = context_window

    return model


def setup_galileo_environment(secrets: Dict[str, Any], console_url: str = "https://console.hp.galileocloud.io/") -> None:
    """
    Configure environment variables for Galileo services (deprecated).

    Args:
        secrets: Dictionary containing API keys (ignored).
        console_url: URL for the Galileo console (ignored).

    Note:
        This function is deprecated. Galileo dependencies have been removed for public release.
    """
    print("⚠️  Warning: Galileo environment setup is disabled - dependencies removed for public release")
    pass


def initialize_galileo_protect(project_name: str, stage_name: Optional[str] = None) -> Tuple[Any, str, str]:
    """
    Initialize Galileo Protect project and stage (deprecated).

    Args:
        project_name: Name for the Galileo Protect project (ignored).
        stage_name: Optional name for the stage (ignored).

    Returns:
        Tuple containing (None, empty_string, empty_string).

    Note:
        This function is deprecated. Galileo dependencies have been removed for public release.
    """
    print("⚠️  Warning: Galileo Protect initialization is disabled - dependencies removed for public release")
    return None, "", ""


def initialize_galileo_evaluator(project_name: str, scorers: Optional[List] = None):
    """
    Initialize a Galileo Prompt Callback for evaluation (deprecated).

    Args:
        project_name: Name for the evaluation project (ignored).
        scorers: List of scorers to use (ignored).

    Returns:
        None

    Note:
        This function is deprecated. Galileo dependencies have been removed for public release.
    """
    print("⚠️  Warning: Galileo Evaluator initialization is disabled - dependencies removed for public release")
    return None
    

def initialize_galileo_observer(project_name: str):
    """
    Initialize a Galileo Observer for monitoring (deprecated).

    Args:
        project_name: Name for the observation project (ignored).

    Returns:
        None

    Note:
        This function is deprecated. Galileo dependencies have been removed for public release.
    """
    print("⚠️  Warning: Galileo Observer initialization is disabled - dependencies removed for public release")
    return None


def login_huggingface(secrets: Dict[str, Any]) -> None:
    """
    Login to Hugging Face using token from secrets.

    Args:
        secrets: Dictionary containing the Hugging Face token.

    Raises:
        ValueError: If the token is missing.
    """
    from huggingface_hub import login

    token = secrets.get("HUGGINGFACE_API_KEY")
    if not token:
        raise ValueError("❌ Hugging Face token not found in secrets.yaml.")
    
    login(token=token)
    print("✅ Logged into Hugging Face successfully.")


def get_model_context_window(model) -> int:
    """
    Get context window using model identifier and lookup table.
    
    Args:
        model: Any language model object (LlamaCpp, HuggingFace, OpenAI, etc.)
        
    Returns:
        int: The determined context window size in tokens, defaulting to 2048 if detection fails
    """
    # Check if we already stored the context window in the model itself
    if hasattr(model, '_context_window') and model._context_window is not None:
        return model._context_window

    # For LlamaCpp: extract filename from model_path
    if hasattr(model, 'model_path'):
        model_filename = os.path.basename(model.model_path)
        if model_filename in MODEL_CONTEXT_WINDOWS:
            return MODEL_CONTEXT_WINDOWS[model_filename]

    # For HuggingFace models: check repo_id
    if hasattr(model, 'repo_id'):
        if model.repo_id in MODEL_CONTEXT_WINDOWS:
            return MODEL_CONTEXT_WINDOWS[model.repo_id]

    # Fall back to direct n_ctx attribute if available
    if hasattr(model, 'n_ctx'):
        return model.n_ctx

    # Check model_kwargs for context window parameters
    if hasattr(model, 'model_kwargs'):
        kwargs = model.model_kwargs
        for param_name in ['n_ctx', 'max_tokens', 'max_length', 'context_window']:
            if param_name in kwargs and kwargs[param_name] is not None:
                return kwargs[param_name]

    # For HuggingFace pipeline models: check tokenizer
    if hasattr(model, 'pipeline') and hasattr(model.pipeline, 'tokenizer') and hasattr(model.pipeline.tokenizer, 'model_max_length'):
        if model.pipeline.tokenizer.model_max_length > 0 and model.pipeline.tokenizer.model_max_length < 1000000000000000:
            return model.pipeline.tokenizer.model_max_length

    # Use a very conservative default if all detection methods fail
    return 2048


def get_context_window(model) -> int:
    """
    Get context window size from model.
    
    Args:
        model: Any language model object
        
    Returns:
        int: The context window size in tokens
    """
    if hasattr(model, '_context_window') and model._context_window is not None:
        return model._context_window

    # Fall back to detection logic
    return get_model_context_window(model)
