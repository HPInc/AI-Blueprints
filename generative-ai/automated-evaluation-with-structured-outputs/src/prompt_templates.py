"""
Prompt templates for automated evaluation using Meta Llama 3.1 8B model.

This module provides prompt templates for automated evaluation tasks,
specifically optimized for the Meta-Llama-3.1-8B-Instruct-Q8_0.gguf model.
"""

from typing import List, Optional


def format_llama_prompt(base_prompt: str) -> str:
    """
    Format prompt template for the Meta Llama 3.1 8B model.
    
    This function applies the correct prompt format structure for the 
    Meta-Llama-3.1-8B-Instruct-Q8_0.gguf model used in this project.
    
    Args:
        base_prompt: The base prompt content to be formatted
        
    Returns:
        str: Properly formatted prompt template for Meta Llama 3.1 8B
    """
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that provides accurate and precise responses for evaluation tasks.<|eot_id|><|start_header_id|>user<|end_header_id|>

{base_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def format_evaluation_prompt(
    key_column: str,
    criteria: List[str],
    context_placeholder: str = "{context}"
) -> str:
    """
    Create a properly formatted evaluation prompt template.
    
    Args:
        key_column: The name of the key column (e.g., "BoothNumber")
        criteria: List of evaluation criteria (e.g., ["Originality", "ScientificRigor"])
        context_placeholder: Placeholder for the context text (default: "{context}")
        
    Returns:
        str: Formatted prompt template for evaluation
    """
    # Build the bullet list dynamically
    criteria_bullets = "\n".join(f"- {c}" for c in criteria)
    
    # Build the example-object fields (all with dummy value 7)
    example_fields = ", ".join(f'"{c}": 7' for c in criteria)
    
    base_prompt = (
        "You are an expert evaluator. "
        f"For each input record, score 1–10 on these criteria:\n"
        f"{criteria_bullets}\n\n"
        "Respond *only* with a valid JSON object of the form:\n"
        "{\n"
        '  "results": [\n'
        f'    {{ "{key_column}": "...", {example_fields} }}\n'
        "  ]\n"
        "}\n"
        "Do not include any other text, explanation, or markup.\n\n"
        f"{context_placeholder}"
    )

    return format_llama_prompt(base_prompt)


def get_evaluation_prompt(
    model_source: str = "local",  # Kept for backward compatibility but not used
    key_column: str = "BoothNumber",
    criteria: List[str] = None
) -> str:
    """
    Get the evaluation prompt template for Meta Llama 3.1 8B model.
    
    Args:
        model_source: Legacy parameter for backward compatibility (ignored)
        key_column: The name of the key column (default: "BoothNumber")
        criteria: List of evaluation criteria (default: ["Originality", "ScientificRigor", "Clarity", "Relevance", "Feasibility", "Brevity"])
        
    Returns:
        str: Formatted prompt template for evaluation
    """
    if criteria is None:
        criteria = ["Originality", "ScientificRigor", "Clarity", "Relevance", "Feasibility", "Brevity"]
    
    return format_evaluation_prompt(key_column, criteria)


def format_isef_evaluation_prompt(
    key_column: str = "ID",
    context_placeholder: str = "{context}"
) -> str:
    """
    Create a properly formatted ISEF evaluation prompt template.
    
    Args:
        key_column: The name of the key column (default: "ID")
        context_placeholder: Placeholder for the context text (default: "{context}")
        
    Returns:
        str: Formatted prompt template for ISEF evaluation
    """
    base_prompt = (
        "You are an expert ISEF (International Science and Engineering Fair) judge. "
        "For each project abstract, evaluate and score 1-10 on the following criteria:\n"
        "- Innovation: How novel and creative is the approach?\n"
        "- Scientific Method: How well does the project follow scientific methodology?\n"
        "- Clarity: How clearly is the project described?\n"
        "- Significance: How important is the potential impact?\n"
        "- Feasibility: How realistic is the project implementation?\n\n"
        "Respond *only* with a valid JSON object of the form:\n"
        "{\n"
        '  "results": [\n'
        f'    {{ "{key_column}": "...", "Innovation": 7, "ScientificMethod": 8, "Clarity": 6, "Significance": 9, "Feasibility": 7 }}\n'
        "  ]\n"
        "}\n"
        "Do not include any other text, explanation, or markup.\n\n"
        f"{context_placeholder}"
    )

    return format_llama_prompt(base_prompt)


def get_isef_evaluation_prompt(
    model_source: str = "local",  # Kept for backward compatibility but not used
    key_column: str = "ID"
) -> str:
    """
    Get the ISEF evaluation prompt template for Meta Llama 3.1 8B model.
    
    Args:
        model_source: Legacy parameter for backward compatibility (ignored)
        key_column: The name of the key column (default: "ID")
        
    Returns:
        str: Formatted prompt template for ISEF evaluation
    """
    return format_isef_evaluation_prompt(key_column)


def get_custom_evaluation_prompt(
    model_source: str,  # Kept for backward compatibility but not used
    custom_instructions: str,
    key_column: str = "ID",
    context_placeholder: str = "{context}"
) -> str:
    """
    Create a custom evaluation prompt with specific instructions.
    
    Args:
        model_source: Legacy parameter for backward compatibility (ignored)
        custom_instructions: Custom instructions for the evaluation task
        key_column: The name of the key column (default: "ID")
        context_placeholder: Placeholder for the context text (default: "{context}")
        
    Returns:
        str: Formatted prompt template for custom evaluation
    """
    base_prompt = f"""You are an expert evaluator.

{custom_instructions}

{context_placeholder}"""

    return format_llama_prompt(base_prompt)
