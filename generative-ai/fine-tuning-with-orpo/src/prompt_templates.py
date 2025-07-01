"""
Prompt templates for fine-tuning with ORPO with model-specific formatting.

This module provides prompt templates for fine-tuning evaluation and comparison tasks,
with specialized handling for different model sources to prevent hallucination.
"""

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

You are a helpful assistant that provides accurate and helpful responses.<|eot_id|><|start_header_id|>user<|end_header_id|>

{base_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    elif model_source == "hugging-face-local":
        # For local HuggingFace models (Llama 3.2 3B) - uses Llama 3.2 chat template with special tokens
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that provides accurate and helpful responses.<|eot_id|><|start_header_id|>user<|end_header_id|>

{base_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    elif model_source == "hugging-face-cloud":
        # For cloud HuggingFace models (Mistral 7B) - uses Mistral's instruction format
        return f"""[INST] {base_prompt} [/INST]"""
    
    else:
        # Fallback to simple format for unknown model sources
        return base_prompt


def format_coding_assistance_prompt(model_source: str, context_placeholder: str = "{prompt}") -> str:
    """
    Create a properly formatted coding assistance prompt template for different model sources.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        context_placeholder: Placeholder for the prompt text (default: "{prompt}")
        
    Returns:
        str: Formatted prompt template for coding assistance
    """
    base_prompt = f"""You are an expert software developer and programmer. Please provide clear, accurate, and helpful coding assistance.

{context_placeholder}

Please provide a comprehensive solution with explanations."""

    return get_prompt_template_for_model_source(model_source, base_prompt)


def format_advice_prompt(model_source: str, context_placeholder: str = "{prompt}") -> str:
    """
    Create a properly formatted advice prompt template for different model sources.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        context_placeholder: Placeholder for the prompt text (default: "{prompt}")
        
    Returns:
        str: Formatted prompt template for general advice
    """
    base_prompt = f"""You are a knowledgeable advisor who provides practical, actionable advice based on best practices and experience.

{context_placeholder}

Please provide thoughtful and helpful advice."""

    return get_prompt_template_for_model_source(model_source, base_prompt)


def format_problem_solving_prompt(model_source: str, context_placeholder: str = "{prompt}") -> str:
    """
    Create a properly formatted problem-solving prompt template for different model sources.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        context_placeholder: Placeholder for the prompt text (default: "{prompt}")
        
    Returns:
        str: Formatted prompt template for problem-solving tasks
    """
    base_prompt = f"""You are an expert problem solver who can analyze complex issues and provide practical, innovative solutions.

{context_placeholder}

Please provide a detailed solution with clear steps and reasoning."""

    return get_prompt_template_for_model_source(model_source, base_prompt)


def format_creative_writing_prompt(model_source: str, context_placeholder: str = "{prompt}") -> str:
    """
    Create a properly formatted creative writing prompt template for different model sources.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        context_placeholder: Placeholder for the prompt text (default: "{prompt}")
        
    Returns:
        str: Formatted prompt template for creative writing tasks
    """
    base_prompt = f"""You are a skilled writer who can create compelling, thoughtful, and well-structured content.

{context_placeholder}

Please create a well-written response that is engaging and appropriate for the task."""

    return get_prompt_template_for_model_source(model_source, base_prompt)


def format_evaluation_prompt(model_source: str, context_placeholder: str = "{prompt}") -> str:
    """
    Create a properly formatted evaluation prompt template for different model sources.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        context_placeholder: Placeholder for the prompt text (default: "{prompt}")
        
    Returns:
        str: Formatted prompt template for evaluation tasks
    """
    base_prompt = f"""You are an expert evaluator who provides thorough, fair, and insightful analysis.

{context_placeholder}

Please provide a comprehensive evaluation with clear reasoning."""

    return get_prompt_template_for_model_source(model_source, base_prompt)


def get_coding_assistance_prompt(model_source: str = "local") -> str:
    """
    Get the coding assistance prompt template formatted for the specified model source.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        
    Returns:
        str: Formatted prompt template for coding assistance
    """
    return format_coding_assistance_prompt(model_source)


def get_advice_prompt(model_source: str = "local") -> str:
    """
    Get the advice prompt template formatted for the specified model source.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        
    Returns:
        str: Formatted prompt template for general advice
    """
    return format_advice_prompt(model_source)


def get_problem_solving_prompt(model_source: str = "local") -> str:
    """
    Get the problem-solving prompt template formatted for the specified model source.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        
    Returns:
        str: Formatted prompt template for problem-solving tasks
    """
    return format_problem_solving_prompt(model_source)


def get_creative_writing_prompt(model_source: str = "local") -> str:
    """
    Get the creative writing prompt template formatted for the specified model source.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        
    Returns:
        str: Formatted prompt template for creative writing tasks
    """
    return format_creative_writing_prompt(model_source)


def get_evaluation_prompt(model_source: str = "local") -> str:
    """
    Get the evaluation prompt template formatted for the specified model source.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        
    Returns:
        str: Formatted prompt template for evaluation tasks
    """
    return format_evaluation_prompt(model_source)


def get_custom_prompt(
    model_source: str,
    custom_instructions: str,
    context_placeholder: str = "{prompt}"
) -> str:
    """
    Create a custom prompt with specific instructions.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        custom_instructions: Custom instructions for the task
        context_placeholder: Placeholder for the prompt text (default: "{prompt}")
        
    Returns:
        str: Formatted prompt template for custom tasks
    """
    base_prompt = f"""{custom_instructions}

{context_placeholder}"""

    return get_prompt_template_for_model_source(model_source, base_prompt)
