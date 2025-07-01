"""
Prompt templates for vanilla RAG with model-specific formatting.

This module provides improved prompt templates for RAG applications,
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
        # For local HuggingFace models (Llama 3.2 3B) - uses Llama 3.2 chat template with special tokens
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that provides accurate and concise responses.<|eot_id|><|start_header_id|>user<|end_header_id|>

{base_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    elif model_source == "hugging-face-cloud":
        # For cloud HuggingFace models (Mistral 7B) - uses Mistral's instruction format
        return f"""[INST] {base_prompt} [/INST]"""
    
    else:
        # Fallback to simple format for unknown model sources
        return base_prompt


# Base template for RAG chatbot
RAG_CHATBOT_TEMPLATE = """You are a chatbot assistant for a Data Science platform created by HP, called 'Z by HP AI Studio'. 
Do not hallucinate and answer questions only if they are related to 'Z by HP AI Studio'. 
Now, answer the question perfectly based on the following context:

{context}

Question: {query}"""


def get_rag_chatbot_prompt(model_source: str = "local") -> ChatPromptTemplate:
    """
    Get the RAG chatbot prompt template formatted for the specified model source.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        
    Returns:
        ChatPromptTemplate for RAG chatbot
    """
    formatted_template = get_prompt_template_for_model_source(model_source, RAG_CHATBOT_TEMPLATE)
    return ChatPromptTemplate.from_template(formatted_template)


def format_rag_chatbot_prompt(model_source: str, context_placeholder: str = "{context}", query_placeholder: str = "{query}") -> str:
    """
    Create a properly formatted RAG chatbot prompt template for different model sources.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        context_placeholder: Placeholder for the context text (default: "{context}")
        query_placeholder: Placeholder for the user query (default: "{query}")
        
    Returns:
        str: Formatted prompt template for RAG chatbot
    """
    base_prompt = f"""You are a chatbot assistant for a Data Science platform created by HP, called 'Z by HP AI Studio'. 
Do not hallucinate and answer questions only if they are related to 'Z by HP AI Studio'. 
Now, answer the question perfectly based on the following context:

{context_placeholder}

Question: {query_placeholder}"""

    return get_prompt_template_for_model_source(model_source, base_prompt)


def get_custom_rag_prompt(
    model_source: str,
    system_message: str,
    context_placeholder: str = "{context}",
    query_placeholder: str = "{query}"
) -> ChatPromptTemplate:
    """
    Create a custom RAG prompt with specific system instructions.
    
    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")
        system_message: Custom system message for the RAG application
        context_placeholder: Placeholder for the context text (default: "{context}")
        query_placeholder: Placeholder for the user query (default: "{query}")
        
    Returns:
        ChatPromptTemplate for custom RAG application
    """
    base_prompt = f"""{system_message}

{context_placeholder}

Question: {query_placeholder}"""

    formatted_template = get_prompt_template_for_model_source(model_source, base_prompt)
    return ChatPromptTemplate.from_template(formatted_template)
