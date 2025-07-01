"""
Prompt templates for text summarization with model-specific formatting.

This module provides improved prompt templates for text summarization,
with specialized handling for different model sources to prevent hallucination.
"""

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from typing import Dict, List, Any, Optional


def get_prompt_template_for_model_source(model_source: str, base_prompt: str) -> str:
    """
    Format prompt template based on model source to follow the expected format for each model.
    
    This function prevents model hallucination by applying the correct prompt format structure
    for different model sources. Each model type has its own expected format.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        base_prompt: The base prompt content to be formatted
        
    Returns:
        str: Properly formatted prompt template for the specified model source
    """
    if model_source == "local":
        # For local Meta Llama 3.1 8B model - uses structured tokenized conversation style
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that provides accurate and concise responses.<|eot_id|><|start_header_id|>user<|end_header_id|>

{base_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    elif model_source == "hugging-face-local":
        # For local HuggingFace models (Llama 3.2 3B) - uses simple role-based format
        return f"""system
You are a helpful assistant that provides accurate and concise responses for text summarization.

user
{base_prompt}

assistant
"""
    
    elif model_source == "hugging-face-cloud":
        # For cloud HuggingFace models (Mistral 7B) - uses Mistral's instruction format
        return f"""[INST] {base_prompt} [/INST]"""
    
    else:
        # Fallback to simple format for unknown model sources
        return base_prompt


# Base template for standard text summarization
SUMMARIZATION_TEMPLATE = """The following text is an excerpt of a text:

###

{context}

###

Please, summarize this text, in a concise and comprehensive manner."""


# Base template for chunk summarization
CHUNK_SUMMARIZATION_TEMPLATE = """The following text is an excerpt of a text

### 
{context} 
###

Please, produce a single paragraph summarizing the given excerpt."""


def get_summarization_prompt(model_source: str = "local") -> PromptTemplate:
    """
    Get the summarization prompt template formatted for the specified model source.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        
    Returns:
        PromptTemplate for text summarization
    """
    formatted_template = get_prompt_template_for_model_source(model_source, SUMMARIZATION_TEMPLATE)
    return PromptTemplate.from_template(formatted_template)


def get_chunk_summarization_prompt(model_source: str = "local") -> PromptTemplate:
    """
    Get the chunk summarization prompt template formatted for the specified model source.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        
    Returns:
        PromptTemplate for chunk summarization
    """
    formatted_template = get_prompt_template_for_model_source(model_source, CHUNK_SUMMARIZATION_TEMPLATE)
    return PromptTemplate.from_template(formatted_template)


def format_summarization_prompt(model_source: str, context_placeholder: str = "{context}") -> str:
    """
    Create a properly formatted summarization prompt template for different model sources.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        context_placeholder: Placeholder for the context text (default: "{context}")
        
    Returns:
        str: Formatted prompt template for text summarization
    """
    base_prompt = f"""The following text is an excerpt of a text:

###

{context_placeholder}

###

Please, summarize this text, in a concise and comprehensive manner."""

    return get_prompt_template_for_model_source(model_source, base_prompt)


def format_chunk_summarization_prompt(model_source: str, context_placeholder: str = "{context}") -> str:
    """
    Create a properly formatted chunk summarization prompt template for different model sources.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        context_placeholder: Placeholder for the context text (default: "{context}")
        
    Returns:
        str: Formatted prompt template for chunk summarization
    """
    base_prompt = f"""The following text is an excerpt of a text

### 
{context_placeholder} 
###

Please, produce a single paragraph summarizing the given excerpt."""

    return get_prompt_template_for_model_source(model_source, base_prompt)


def get_custom_summarization_prompt(
    model_source: str,
    custom_instructions: str,
    context_placeholder: str = "{context}"
) -> PromptTemplate:
    """
    Create a custom summarization prompt with specific instructions.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        custom_instructions: Custom instructions for the summarization task
        context_placeholder: Placeholder for the context text (default: "{context}")
        
    Returns:
        PromptTemplate for custom summarization
    """
    base_prompt = f"""The following text is an excerpt of a text:

###

{context_placeholder}

###

{custom_instructions}"""

    formatted_template = get_prompt_template_for_model_source(model_source, base_prompt)
    return PromptTemplate.from_template(formatted_template)
