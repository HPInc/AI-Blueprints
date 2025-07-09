import re
from typing import List

def estimate_token_count(text: str) -> int:
    return len(text) // 4  # Approximation: 4 characters per token

def split_by_top_level_headers(markdown: str) -> List[str]:
    matches = list(re.finditer(r'^\s*#{1,6}\s.*', markdown, flags=re.MULTILINE))
    if not matches:
        return [markdown]

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        sections.append(markdown[start:end])

    return sections

def smart_sentence_split(text: str) -> List[str]:
    """
    Split text at sentence boundaries or placeholder boundaries.
    This avoids variable-width lookbehind issues in Python.
    """
    pattern = r'([.!?]\s+(?=[A-Z]))|(__PLACEHOLDER\d+__)'
    matches = re.split(pattern, text)

    # Reconstruct full sentences while preserving split points
    parts = []
    buffer = ''
    for chunk in matches:
        if chunk is None:
            continue
        buffer += chunk
        if re.match(r'[.!?]\s+$', chunk) or re.match(r'__PLACEHOLDER\d+__', chunk.strip()):
            parts.append(buffer.strip())
            buffer = ''
    if buffer.strip():
        parts.append(buffer.strip())

    return parts

def chunk_large_section(section: str, max_tokens: int = 1000) -> List[str]:
    """
    Chunk a section while avoiding breaks mid-sentence or mid-placeholder.
    """
    chunks = []
    current_chunk = []
    current_token_count = 0

    parts = smart_sentence_split(section)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        part_token_count = estimate_token_count(part)

        # If adding this part would exceed max tokens, finalize current chunk
        if current_token_count + part_token_count > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk).strip())
                current_chunk = []
                current_token_count = 0

        current_chunk.append(part)
        current_token_count += part_token_count

    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())

    return chunks

def chunk_markdown(markdown: str, max_tokens: int = 1000) -> List[str]:
    sections = split_by_top_level_headers(markdown)
    final_chunks = []

    for section in sections:
        if estimate_token_count(section) <= max_tokens:
            final_chunks.append(section.strip())
        else:
            final_chunks.extend(chunk_large_section(section, max_tokens=max_tokens))

    return final_chunks
