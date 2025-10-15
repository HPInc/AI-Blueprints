"""
Prompt templates for text generation with model-specific formatting.

This module provides prompt templates for scientific paper analysis and text generation,
with model-specific formatting to prevent hallucination.
"""

from langchain.prompts import PromptTemplate
from typing import Optional


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

You are a helpful assistant that provides accurate and concise responses for scientific paper analysis and text generation.<|eot_id|><|start_header_id|>user<|end_header_id|>

{base_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    elif model_source == "hugging-face-local":
        # For local HuggingFace models (Llama 3.2 3B) - uses Llama 3.2 chat template with special tokens
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that provides accurate and concise responses for scientific paper analysis and text generation.<|eot_id|><|start_header_id|>user<|end_header_id|>

{base_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    elif model_source == "hugging-face-cloud":
        # For cloud HuggingFace models (Mistral 7B) - uses Mistral's instruction format
        return f"""[INST] {base_prompt} [/INST]"""

    else:
        # Fallback to simple format for unknown model sources
        return base_prompt


# Base template for scientific paper analysis
SCIENTIFIC_PAPER_ANALYSIS_TEMPLATE = """You are analyzing the following scientific paper:

{context}

Instruction: {prompt}

"""


def format_scientific_paper_analysis_prompt(
    model_source: str = "local",
) -> PromptTemplate:
    """
    Create a properly formatted scientific paper analysis prompt template for different model sources.

    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")

    Returns:
        PromptTemplate: Formatted prompt template for scientific paper analysis
    """
    formatted_template = get_prompt_template_for_model_source(
        model_source, SCIENTIFIC_PAPER_ANALYSIS_TEMPLATE
    )
    return PromptTemplate.from_template(formatted_template)


def get_scientific_paper_analysis_prompt(model_source: str = "local") -> PromptTemplate:
    """
    Get the scientific paper analysis prompt template formatted for the specified model source.

    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")

    Returns:
        PromptTemplate for scientific paper analysis
    """
    return format_scientific_paper_analysis_prompt(model_source)


# Base template for general text generation
TEXT_GENERATION_TEMPLATE = """You are a helpful text generation assistant.

Context: {context}

Task: {prompt}

Please provide a comprehensive and accurate response based on the context and task provided.
"""


def format_text_generation_prompt(model_source: str = "local") -> PromptTemplate:
    """
    Create a properly formatted text generation prompt template for different model sources.

    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")

    Returns:
        PromptTemplate: Formatted prompt template for text generation
    """
    formatted_template = get_prompt_template_for_model_source(
        model_source, TEXT_GENERATION_TEMPLATE
    )
    return PromptTemplate.from_template(formatted_template)


def get_text_generation_prompt(model_source: str = "local") -> PromptTemplate:
    """
    Get the text generation prompt template formatted for the specified model source.

    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")

    Returns:
        PromptTemplate for text generation
    """
    return format_text_generation_prompt(model_source)
