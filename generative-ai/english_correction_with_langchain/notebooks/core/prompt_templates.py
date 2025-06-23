"""
Prompt templates for markdown grammar correction.

This module provides structured prompt templates to guide language models in refining markdown content.
Special emphasis is placed on preserving formatting, placeholders, and document structure.
"""

from langchain.prompts import PromptTemplate  
from typing import Dict, List, Any

# Template to correct English grammar in markdown content
MARKDOWN_CORRECTION_TEMPLATE = """
You are an English grammar assistant.

Your task is to **correct grammar and improve clarity** in the following Markdown content, while preserving its meaning and Markdown formatting.

Return **only** the corrected Markdown. Do not add any explanations or extra text.

---

Markdown input:
{markdown}

---

Corrected Markdown:
"""

def get_markdown_correction_prompt() -> PromptTemplate:
    """
    Get the markdown correction prompt template.

    Returns:
        PromptTemplate for markdown correction
    """
    return PromptTemplate.from_template(MARKDOWN_CORRECTION_TEMPLATE)
