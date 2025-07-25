import os
import yaml
import importlib.util
import multiprocessing
from typing import Dict, Any, Optional, Union, List, Tuple

#Default models to be loaded in our examples:
DEFAULT_MODELS = {
    "local": "/home/jovyan/datafabric/llama2-7b/ggml-model-f16-Q5_K_M.gguf",
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
}

def load_config_and_secrets(
    config_path: str = "../../configs/config.yaml",
    secrets_path: str = "../../configs/secrets.yaml"
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

        pipe = pipeline("text-generation", model=hf_model, tokenizer=tokenizer, max_new_tokens=100, device=0)
        model = HuggingFacePipeline(pipeline=pipe)
        
    elif model_source == "tensorrt":
        #If a Hugging Face model is specified, it will be used - otherwise, it will try loading the model from local_path
        try:
            import tensorrt_llm
            sampling_params = tensorrt_llm.SamplingParams(temperature=0.1, top_p=0.95, max_tokens=512) 
            if hf_repo_id != "":
                return TensorRTLangchain(model_path = hf_repo_id, sampling_params = sampling_params)
            else:
                model_config = os.path.join(local_model_path, config.json)
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
            n_gpu_layers=-1, # Number of GPU-loaded layers can be reduced if there is insufficient memory (-1 assigns all layers to GPU)                            
            n_batch=512,                                 
            n_ctx=32000,
            max_tokens=1024,
            f16_kv=True,
            use_mmap=False,                             
            low_vram=False,                            
            rope_scaling=None,
            temperature=0.0,
            repeat_penalty=1.0,
            streaming=False,
            stop=None,
            seed=42,
            num_threads=multiprocessing.cpu_count(),
            verbose=False #
        )
    else:
        raise ValueError(f"Unsupported model source: {model_source}")

    # Store context window as model attribute for easy access
    if model and hasattr(model, '__dict__'):
        model.__dict__['_context_window'] = context_window

    return model