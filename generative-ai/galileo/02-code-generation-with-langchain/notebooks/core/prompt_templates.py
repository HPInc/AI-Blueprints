"""
Prompt templates for code repository RAG.

This module provides improved prompt templates for code repository analysis,
with specialized handling for different question types and multi-document context.
"""

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from typing import Dict, List, Any

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

def get_dynamic_repository_prompt() -> ChatPromptTemplate:
    """
    Get the dynamic repository prompt template.
    
    Returns:
        ChatPromptTemplate for repository questions
    """
    return ChatPromptTemplate.from_template(DYNAMIC_REPOSITORY_TEMPLATE)

def get_code_description_prompt() -> ChatPromptTemplate:
    """
    Get the code description prompt template.
    
    Returns:
        ChatPromptTemplate for code description
    """
    return ChatPromptTemplate.from_template(CODE_DESCRIPTION_TEMPLATE)

def get_code_generation_prompt() -> ChatPromptTemplate:
    """
    Get the code generation prompt template.
    
    Returns:
        ChatPromptTemplate for code generation
    """
    return ChatPromptTemplate.from_template(CODE_GENERATION_TEMPLATE)

def get_metadata_generation_prompt() -> PromptTemplate:
    """
    Get the metadata generation prompt template.
    
    Returns:
        PromptTemplate for metadata generation
    """
    return PromptTemplate.from_template(METADATA_GENERATION_TEMPLATE)

def get_specialized_prompt(question_types: List[str]) -> ChatPromptTemplate:
    """
    Get a specialized prompt based on the detected question type.
    
    Args:
        question_types: List of identified question types
        
    Returns:
        Specialized ChatPromptTemplate or default if no specialization
    """
    if not question_types:
        return get_code_description_prompt()
        
    # Get the primary question type (highest confidence)
    primary_type = question_types[0]
    
    if primary_type == "dependency":
        # Specialized prompt for dependency questions
        template = """You are a dependency analysis expert for code repositories.

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
Organize your response to clearly indicate primary vs. optional dependencies.
"""
        return ChatPromptTemplate.from_template(template)
        
    elif primary_type == "implementation":
        # Specialized prompt for implementation questions
        template = """You are an expert code analyst with deep understanding of software implementation patterns.

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
Use code examples from the context when relevant to illustrate key points.
"""
        return ChatPromptTemplate.from_template(template)
        
    elif primary_type == "error":
        # Specialized prompt for error questions
        template = """You are a debugging expert who specializes in identifying and resolving code issues.

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
When suggesting fixes, be specific and provide code examples if possible.
"""
        return ChatPromptTemplate.from_template(template)
    
    # Default to the standard code description prompt
    return get_code_description_prompt()
