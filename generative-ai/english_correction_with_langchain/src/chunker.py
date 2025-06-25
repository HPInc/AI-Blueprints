import re
from typing import List


def estimate_token_count(text: str) -> int:
    """
    Estimate token count based on average 4 characters per token (OpenAI's rule of thumb).
    
     Args:
        text (str): The text string to estimate token count for.

    Returns:
        int: Approximate number of tokens in the input text.
    """
    return len(text) // 4


def split_by_top_level_headers(markdown: str) -> List[str]:
    """
    Splits a markdown file by top-level headers (# ) while preserving content.

    Args:
        markdown (str): The full markdown text to be split.

    Returns:
        List[str]: List of markdown sections starting with a top-level header.
    """
    # Include leading header line in each section
    sections = re.split(r'(?=^#\s)', markdown, flags=re.MULTILINE)
    return [section.strip() for section in sections if section.strip()]


def chunk_large_section(section: str, max_tokens: int = 1000) -> List[str]:
    """
    Chunk large markdown sections without changing structure.

    Args:
        section (str): A section of markdown text to be chunked.
        max_tokens (int, optional): Maximum number of tokens per chunk. Defaults to 1000.

    Returns:
        List[str]: List of markdown chunks that do not exceed the max token limit.
    """
    lines = section.splitlines()
    chunks = []
    current_chunk = []
    current_token_count = 0

    for line in lines:
        line_token_estimate = estimate_token_count(line)
        if current_token_count + line_token_estimate > max_tokens:
            # Save current chunk
            chunks.append("\n".join(current_chunk).strip())
            current_chunk = []
            current_token_count = 0

        current_chunk.append(line)
        current_token_count += line_token_estimate

    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())

    return chunks


def chunk_markdown(markdown: str, max_tokens: int = 1000) -> List[str]:
    """
    Chunk markdown into manageable parts for LLM input, preserving formatting.
    
    Split by top-level headers (`# `) and sub-chunk if any section is too large

    Args:
        markdown (str): Raw markdown content to chunk.
        max_tokens (int, optional): Token limit per chunk. Defaults to 1000.

    Returns:
        List[str]: List of markdown chunks each under the max token threshold.
    """
    sections = split_by_top_level_headers(markdown)
    final_chunks = []

    for section in sections:
        if estimate_token_count(section) <= max_tokens:
            final_chunks.append(section)
        else:
            final_chunks.extend(chunk_large_section(section, max_tokens=max_tokens))

    return final_chunks
