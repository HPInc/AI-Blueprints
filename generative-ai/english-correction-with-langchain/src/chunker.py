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

def chunk_markdown(markdown: str, max_tokens: int = 100) -> List[str]:
    sections = split_by_top_level_headers(markdown)
    final_chunks = []

    for section in sections:
        if estimate_token_count(section) <= max_tokens:
            final_chunks.append(section.strip())
        else:
            final_chunks.extend(chunk_large_section(section, max_tokens=max_tokens))

    # Post-process chunks: split any that exceed 1000 characters
    def split_long_chunk(chunk: str, max_chars: int = 3500) -> List[str]:
        if len(chunk) <= max_chars:
            return [chunk]

        # Try splitting on newlines first
        parts = re.split(r'(?<=\n)', chunk)
        subchunks = []
        buffer = ""

        for part in parts:
            if len(buffer) + len(part) > max_chars:
                if buffer:
                    subchunks.append(buffer.strip())
                buffer = part
            else:
                buffer += part

        if buffer.strip():
            subchunks.append(buffer.strip())

        # If still too long, split on sentence boundary
        final_subchunks = []
        for sub in subchunks:
            if len(sub) <= max_chars:
                final_subchunks.append(sub)
            else:
                sentences = re.split(r'(?<=[.!?])\s+', sub)
                sentence_buffer = ""
                for s in sentences:
                    if len(sentence_buffer) + len(s) > max_chars:
                        final_subchunks.append(sentence_buffer.strip())
                        sentence_buffer = s
                    else:
                        sentence_buffer += (" " if sentence_buffer else "") + s
                if sentence_buffer.strip():
                    final_subchunks.append(sentence_buffer.strip())

        return final_subchunks

    # Apply the splitting to each chunk
    adjusted_chunks = []
    for chunk in final_chunks:
        adjusted_chunks.extend(split_long_chunk(chunk))

    return adjusted_chunks