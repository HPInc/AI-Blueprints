"""
Utility functions for the Text Summarization GenAI Blueprint.
Provides model initialization, configuration management, and helper functions.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default model configurations
DEFAULT_MODELS = {
    "local": "/home/jovyan/local/datafabric/meta-llama3.1-8b-Q8.gguf",
    "hugging-face-cloud": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "hugging-face-local": "meta-llama/Meta-Llama-3.1-8B-Instruct"
}

def configure_hf_cache(cache_dir: str = "/home/jovyan/local/hugging_face") -> None:
    """
    Configure Hugging Face cache directory.
    
    Args:
        cache_dir: Directory path for Hugging Face cache
    """
    # Ensure cache directory exists
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Set environment variables for Hugging Face
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    
    logger.info(f"✅ Hugging Face cache configured at: {cache_dir}")


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
        Tuple containing (config_dict, secrets_dict)
        
    Raises:
        FileNotFoundError: If either file is not found.
    """
    # Load configuration file
    if not Path(config_path).exists():
        raise FileNotFoundError(f"config.yaml file not found in path: {config_path}")
    
    # Load secrets file
    if not Path(secrets_path).exists():
        raise FileNotFoundError(f"secrets.yaml file not found in path: {secrets_path}")
        
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file) or {}
        
    with open(secrets_path, 'r') as file:
        secrets = yaml.safe_load(file) or {}
        
    logger.info(f"✅ Configuration loaded from: {config_path}")
    logger.info(f"✅ Secrets loaded from: {secrets_path}")
    
    return config, secrets


def configure_proxy(config: Dict[str, Any]) -> None:
    """
    Configure proxy settings from configuration.
    
    Args:
        config: Configuration dictionary that may contain proxy settings.
    """
    proxy = config.get("proxy")
    if proxy:
        os.environ["HTTPS_PROXY"] = proxy
        os.environ["HTTP_PROXY"] = proxy
        logger.info(f"✅ Proxy configured: {proxy}")
    else:
        logger.info("ℹ️  No proxy configuration found.")


def initialize_llm(
    model_source: str = "local",
    secrets: Optional[Dict[str, Any]] = None,
    local_model_path: str = DEFAULT_MODELS["local"],
    hf_repo_id: str = ""
) -> Any:
    """
    Initialize a language model based on the specified source.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-cloud", "hugging-face-local")
        secrets: Dictionary containing API keys and tokens
        local_model_path: Path to local model file
        hf_repo_id: Hugging Face repository ID
        
    Returns:
        Initialized language model object
        
    Raises:
        ValueError: If model source is unsupported or model file not found
        ImportError: If required packages are not installed
    """
    context_window = 8192  # Default context window
    
    try:
        if model_source == "local":
            # Use local model with llama-cpp-python
            try:
                from langchain_community.llms import LlamaCpp
                from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
            except ImportError:
                raise ImportError("langchain-community is required for local models. Install it with: pip install langchain-community")
            
            # Verify model file exists
            if not Path(local_model_path).exists():
                raise FileNotFoundError(f"Local model file not found: {local_model_path}")
            
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            model = LlamaCpp(
                model_path=local_model_path,
                temperature=0.2,
                max_tokens=8192,
                n_ctx=context_window,
                callback_manager=callback_manager,
                verbose=False,
                stop=[],
                streaming=False
            )
            
            logger.info(f"✅ Local model loaded from: {local_model_path}")
            
        elif model_source == "hugging-face-cloud":
            # Use Hugging Face cloud API
            try:
                from langchain_huggingface import HuggingFaceEndpoint
            except ImportError:
                raise ImportError("langchain-huggingface is required for HF cloud models. Install it with: pip install langchain-huggingface")
            
            if not secrets or "HUGGINGFACE_API_KEY" not in secrets:
                raise ValueError("HUGGINGFACE_API_KEY is required for cloud models")
            
            model = HuggingFaceEndpoint(
                repo_id=hf_repo_id or DEFAULT_MODELS["hugging-face-cloud"],
                max_length=8192,
                temperature=0.2,
                huggingfacehub_api_token=secrets["HUGGINGFACE_API_KEY"]
            )
            
            logger.info(f"✅ Hugging Face cloud model loaded: {hf_repo_id or DEFAULT_MODELS['hugging-face-cloud']}")
            
        elif model_source == "hugging-face-local":
            # Use local Hugging Face pipeline
            try:
                from langchain_huggingface import HuggingFacePipeline
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                import torch
            except ImportError:
                raise ImportError("transformers and torch are required for local HF models. Install them with: pip install transformers torch")
            
            # Load model and tokenizer
            model_id = hf_repo_id or DEFAULT_MODELS["hugging-face-local"]
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=hf_model,
                tokenizer=tokenizer,
                max_length=8192,
                temperature=0.2
            )
            
            model = HuggingFacePipeline(pipeline=pipe)
            
            logger.info(f"✅ Local Hugging Face model loaded: {model_id}")
            
        else:
            raise ValueError(f"Unsupported model source: {model_source}. Supported options: 'local', 'hugging-face-cloud', 'hugging-face-local'")
        
        # Store context window as model attribute for easy access
        if model and hasattr(model, '__dict__'):
            model.__dict__['_context_window'] = context_window
            
        return model
        
    except Exception as e:
        logger.error(f"❌ Error initializing model: {str(e)}")
        raise


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
    try:
        from huggingface_hub import login
    except ImportError:
        raise ImportError("huggingface_hub is required. Install it with: pip install huggingface_hub")

    token = secrets.get("HUGGINGFACE_API_KEY")
    if not token:
        raise ValueError("❌ Hugging Face token not found in secrets.yaml.")
    
    login(token=token)
    logger.info("✅ Logged into Hugging Face successfully.")


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
    prefixes = ["Answer:", "Expected Answer:", "Expected Output:", "Python code:", "Here's the code:", "My Response:", "Response:"]
    for prefix in prefixes:
        if result.lstrip().startswith(prefix):
            result = result.lstrip()[len(prefix):].lstrip()
            break
    
    # Handle markdown code blocks
    if "```" in result:
        # Extract content between code blocks
        parts = result.split("```")
        if len(parts) >= 3:
            # Get the code part (should be the middle section)
            code_part = parts[1]
            # Remove language identifier if present
            lines = code_part.split('\n')
            if lines and lines[0].strip() in ['python', 'py', 'code']:
                lines = lines[1:]
            result = '\n'.join(lines)
        else:
            # Remove just the backticks
            result = result.replace("```", "")
    
    # Remove extra whitespace and ensure proper formatting
    result = result.strip()
    
    return result


def generate_text_with_retries(chain, example_input, callbacks=None, max_attempts=3, min_text_length=10):
    """
    Generate text with retries to handle potential failures.
    
    Args:
        chain: The LangChain chain to use for generation
        example_input: Input data for the chain
        callbacks: Optional callbacks for the chain
        max_attempts: Maximum number of retry attempts
        min_text_length: Minimum acceptable text length
        
    Returns:
        Generated text string
        
    Raises:
        Exception: If all retry attempts fail
    """
    for attempt in range(max_attempts):
        try:
            if callbacks:
                result = chain.invoke(example_input, config=dict(callbacks=callbacks))
            else:
                result = chain.invoke(example_input)
            
            # Validate the result
            if isinstance(result, str) and len(result.strip()) >= min_text_length:
                logger.info(f"✅ Text generation successful on attempt {attempt + 1}")
                return result.strip()
            else:
                logger.warning(f"⚠️  Generated text too short on attempt {attempt + 1}, retrying...")
                
        except Exception as e:
            logger.warning(f"⚠️  Text generation failed on attempt {attempt + 1}: {str(e)}")
            if attempt == max_attempts - 1:  # Last attempt
                raise e
    
    raise Exception(f"Failed to generate valid text after {max_attempts} attempts")


def get_model_context_window(model) -> int:
    """
    Get the context window size for a model.
    
    Args:
        model: The language model object
        
    Returns:
        Context window size in tokens
    """
    # Try to get stored context window
    if hasattr(model, '__dict__') and '_context_window' in model.__dict__:
        return model.__dict__['_context_window']
    
    # Default fallback
    return 8192


def get_context_window(model) -> int:
    """
    Alias for get_model_context_window for backward compatibility.
    
    Args:
        model: The language model object
        
    Returns:
        Context window size in tokens
    """
    return get_model_context_window(model)


def dynamic_retriever(query: str, collection, top_n: int = None, context_window: int = None) -> List:
    """
    Dynamically retrieve relevant documents based on available context window.
    
    Args:
        query: Search query string
        collection: Vector store collection to search
        top_n: Number of documents to retrieve (if None, calculated dynamically)
        context_window: Available context window (if None, uses default)
        
    Returns:
        List of retrieved documents
    """
    if context_window is None:
        context_window = 8192
    
    if top_n is None:
        # Dynamically calculate based on estimated token usage
        # Assume ~500 tokens per document on average
        # Reserve 25% of context for prompt and response
        available_tokens = int(context_window * 0.75)
        top_n = max(1, available_tokens // 500)
    
    try:
        # Perform similarity search
        results = collection.similarity_search(query, k=top_n)
        logger.info(f"✅ Retrieved {len(results)} documents for query")
        return results
    except Exception as e:
        logger.error(f"❌ Error during document retrieval: {str(e)}")
        return []
