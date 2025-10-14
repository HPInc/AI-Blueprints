"""
Prompt templates for code repository RAG with model-specific formatting.

This module provides improved prompt templates for code repository analysis,
with specialized handling for different question types, multi-document context,
and model-specific formatting to prevent hallucination.
"""

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from typing import Dict, List, Any


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

You are a helpful assistant that provides accurate and concise responses for code repository analysis.<|eot_id|><|start_header_id|>user<|end_header_id|>

{base_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    elif model_source == "hugging-face-local":
        # For local HuggingFace models (Llama 3.2 3B) - uses Llama 3.2 chat template with special tokens
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that provides accurate and concise responses for code repository analysis.<|eot_id|><|start_header_id|>user<|end_header_id|>

{base_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    elif model_source == "hugging-face-cloud":
        # For cloud HuggingFace models (Mistral 7B) - uses Mistral's instruction format
        return f"""[INST] {base_prompt} [/INST]"""

    else:
        # Fallback to simple format for unknown model sources
        return base_prompt


# Template for code repository questions with dynamic multi-document context
DYNAMIC_REPOSITORY_TEMPLATE = """
**Repository Question:** {question}

**Relevant Code Context:**
{context}

**Instructions:**
- Answer the question directly using the provided code context
- Reference specific files and line numbers as evidence when possible
- Synthesize information from multiple sources when relevant
- Explain your reasoning clearly and concisely
- If the context doesn't contain enough information, state what's missing

Please provide a complete and accurate answer to the question based on the context provided.
"""

# Template for code description with multiple files
CODE_DESCRIPTION_TEMPLATE = """You are a code analysis assistant with expert understanding of programming languages and software development.

**User Question:** {question}

**Code Context:**
{context}

Based on the code context above, provide a clear, concise and accurate answer to the user's question.
Focus on the most relevant files and code snippets that directly address the question.
When discussing code:
1. Reference specific file names and functions
2. Explain code purpose and functionality
3. Identify key dependencies and relationships between components
4. Highlight important implementation details

Your answer should synthesize information from all relevant context files and be comprehensive.
"""

# Template for code generation with context
CODE_GENERATION_TEMPLATE = """You are an expert code generator that produces clean, idiomatic, and efficient Python code.

**Task:** {question}

**Relevant Context:**
{context}

Write complete, working Python code that solves the requested task. The code should be:
- Well-structured and organized
- Properly documented with docstrings and comments
- Error-handled appropriately
- Styled according to PEP 8 conventions

Focus on creating a complete, executable solution. Use best practices and standard libraries when appropriate.
Include imports at the top of your code. Do not omit any critical functionality.

Your code:
```python
"""

# Template for metadata generation
METADATA_GENERATION_TEMPLATE = """
You will receive three pieces of information: a code snippet, a file name, and an optional context. Based on this information, explain in a clear, summarized and concise way what the code snippet is doing.

Code:
{code}

File name:
{filename}

Context:
{context}

Describe what the code above does.
"""


def get_dynamic_repository_prompt(model_source: str = "local") -> ChatPromptTemplate:
    """
    Get the dynamic repository prompt template formatted for the specified model source.

    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")

    Returns:
        ChatPromptTemplate for repository questions
    """
    formatted_template = get_prompt_template_for_model_source(
        model_source, DYNAMIC_REPOSITORY_TEMPLATE
    )
    return ChatPromptTemplate.from_template(formatted_template)


def get_code_description_prompt(model_source: str = "local") -> ChatPromptTemplate:
    """
    Get the code description prompt template formatted for the specified model source.

    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")

    Returns:
        ChatPromptTemplate for code description
    """
    formatted_template = get_prompt_template_for_model_source(
        model_source, CODE_DESCRIPTION_TEMPLATE
    )
    return ChatPromptTemplate.from_template(formatted_template)


def get_code_generation_prompt(model_source: str = "local") -> ChatPromptTemplate:
    """
    Get the code generation prompt template formatted for the specified model source.

    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")

    Returns:
        ChatPromptTemplate for code generation
    """
    formatted_template = get_prompt_template_for_model_source(
        model_source, CODE_GENERATION_TEMPLATE
    )
    return ChatPromptTemplate.from_template(formatted_template)


def get_metadata_generation_prompt(model_source: str = "local") -> PromptTemplate:
    """
    Get the metadata generation prompt template formatted for the specified model source.

    Args:
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")

    Returns:
        PromptTemplate for metadata generation
    """
    formatted_template = get_prompt_template_for_model_source(
        model_source, METADATA_GENERATION_TEMPLATE
    )
    return PromptTemplate.from_template(formatted_template)


def get_specialized_prompt(
    question_types: List[str], model_source: str = "local"
) -> ChatPromptTemplate:
    """
    Get a specialized prompt based on the detected question type with model-specific formatting.

    Args:
        question_types: List of identified question types
        model_source: Source of the model ("local", "hugging-face-local", "hugging-face-cloud")

    Returns:
        Specialized ChatPromptTemplate or default if no specialization
    """
    if not question_types:
        return get_code_description_prompt(model_source)

    # Get the primary question type (highest confidence)
    primary_type = question_types[0]

    if primary_type == "dependency":
        # Specialized prompt for dependency questions
        base_template = """You are a dependency analysis expert for code repositories.

**Dependency Question:** {question}

**Repository Context:**
{context}

Analyze the repository context and provide a detailed answer about the dependencies.
Focus specifically on:
1. Required packages, libraries, and modules
2. Version requirements and constraints
3. Installation instructions if available
4. Dependencies between components
5. External vs. internal dependencies

Reference specific configuration files like requirements.txt, setup.py, package.json, etc.
Organize your response to clearly indicate primary vs. optional dependencies."""
        formatted_template = get_prompt_template_for_model_source(
            model_source, base_template
        )
        return ChatPromptTemplate.from_template(formatted_template)

    elif primary_type == "implementation":
        # Specialized prompt for implementation questions
        base_template = """You are an expert code analyst with deep understanding of software implementation patterns.

**Implementation Question:** {question}

**Repository Context:**
{context}

Analyze the implementation details in the repository context and provide a comprehensive answer.
Focus on:
1. Key algorithms and data structures
2. Control flow and architectural patterns
3. Function and class relationships
4. Performance considerations
5. Edge case handling

Reference specific code files, functions, and line numbers to support your explanation.
Use code examples from the context when relevant to illustrate key points."""
        formatted_template = get_prompt_template_for_model_source(
            model_source, base_template
        )
        return ChatPromptTemplate.from_template(formatted_template)

    elif primary_type == "error":
        # Specialized prompt for error questions
        base_template = """You are a debugging expert who specializes in identifying and resolving code issues.

**Error Question:** {question}

**Repository Context:**
{context}

Analyze the code context to identify potential issues, errors, or bugs related to the question.
Focus on:
1. Common error patterns and anti-patterns
2. Exception handling and edge cases
3. Potential fixes or workarounds
4. Root cause analysis
5. Best practices for avoiding similar issues

Reference specific problematic code sections and explain why they might cause issues.
When suggesting fixes, be specific and provide code examples if possible."""
        formatted_template = get_prompt_template_for_model_source(
            model_source, base_template
        )
        return ChatPromptTemplate.from_template(formatted_template)

    # Default to the standard code description prompt
    return get_code_description_prompt(model_source)
