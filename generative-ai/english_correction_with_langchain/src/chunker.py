import re
from typing import List


def estimate_token_count(text: str) -> int:
    return len(text) // 4


def split_by_top_level_headers(markdown: str) -> List[str]:
    """
    Splits a markdown file by top-level headers (# ) while preserving content.

    Returns:
        List[str]: List of markdown sections starting with a top-level header.
    """
    matches = list(re.finditer(r'^#\s.*', markdown, flags=re.MULTILINE))
    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        sections.append(markdown[start:end])  # No stripping
    return sections


def chunk_large_section(section: str, max_tokens: int = 1000) -> List[str]:
    """
    Chunk a section of markdown text exactly (preserving all whitespace, formatting, etc.)

    Args:
        section (str): A markdown section that exceeds max_tokens.
        max_tokens (int): Token limit per chunk.

    Returns:
        List[str]: List of exact markdown chunks under token limit.
    """
    chunks = []
    start = 0
    length = len(section)

    while start < length:
        est_chunk_size = max_tokens * 4
        end = min(start + est_chunk_size, length)

        # Prefer to end on a newline
        newline_pos = section.rfind('\n', start + 1, end)
        if newline_pos > start:
            end = newline_pos + 1  # Include the newline

        chunks.append(section[start:end])
        start = end

    return chunks


def chunk_markdown(markdown: str, max_tokens: int = 1000) -> List[str]:
    """
    Chunk markdown into pieces under the max token threshold,
    while ensuring exact reconstruction is possible.

    Args:
        markdown (str): Raw markdown content to chunk.
        max_tokens (int, optional): Token limit per chunk.

    Returns:
        List[str]: List of markdown chunks.
    """
    sections = split_by_top_level_headers(markdown)
    final_chunks = []

    for section in sections:
        if estimate_token_count(section) <= max_tokens:
            final_chunks.append(section)
        else:
            final_chunks.extend(chunk_large_section(section, max_tokens=max_tokens))

    return final_chunks
