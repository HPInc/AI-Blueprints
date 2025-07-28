import base64
import logging
import time
from functools import wraps
from IPython.display import HTML, display

# Configure logger
logger: logging.Logger = logging.getLogger("AIS_logger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent duplicate logs from parent loggers

# Set formatter
formatter: logging.Formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Configure and attach stream handler
stream_handler: logging.StreamHandler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def log_timing(func):
    """
    Decorator that logs the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds."  +
                   "\n--------------------------------------------------------------\n")
        return result
    return wrapper


def get_response_from_llm(llm, system_prompt, user_prompt):
    meta_llama_prompt = f'''
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    '''
    return llm(meta_llama_prompt)


def display_image(image_bytes: bytes, width: int = 400) -> str:
    """
    Converts image bytes to an HTML string for visualization in Jupyter.

    Args:
        image_bytes (bytes): Raw image content in PNG/JPEG format.
        width (int): Desired width in pixels for display.

    Returns:
        str: HTML <img> tag with base64 image data.
    """
    decoded_img_bytes = base64.b64encode(image_bytes).decode('utf-8')
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
