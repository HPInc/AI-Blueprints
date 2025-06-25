"""
Prompt templates for markdown grammar correction.

This module provides structured prompt templates to guide language models in refining markdown content.
Special emphasis is placed on preserving formatting, placeholders, and document structure.
"""

from langchain.prompts import PromptTemplate  
from typing import Dict, List, Any

# Template to correct English grammar in markdown content
MARKDOWN_CORRECTION_TEMPLATE = """
Fix only grammatical errors in this text. Preserve all formatting exactly. Do not include any additional notes or comments. 

IMPORTANT: Text contains PLACEHOLDER tokens (like __PLACEHOLDER_1__) that represent protected content. Leave ALL placeholders exactly as they are. They must all be present in the output.

Text to correct:
{markdown}

Corrected text:
"""

def get_markdown_correction_prompt() -> PromptTemplate:
    """
    Get the markdown correction prompt template.

    Returns:
        PromptTemplate for markdown correction
    """
    return PromptTemplate.from_template(MARKDOWN_CORRECTION_TEMPLATE)