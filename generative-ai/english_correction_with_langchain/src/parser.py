from typing import Tuple, Dict
import re
from markdown_it import MarkdownIt
from bs4 import BeautifulSoup


def parse_md_for_grammar_correction(md_content: str) -> Tuple[Dict[str, str], str]:
    md = MarkdownIt()
    tokens = md.parse(md_content)

    placeholder_map = {}
    counter = 0

    def get_next_placeholder(value: str, prefix="PLACEHOLDER") -> str:
        nonlocal counter
        counter += 1
        key = f"__{prefix}_{counter}__"
        placeholder_map[key] = value
        return f"<<{key}>>"

    lines = md_content.splitlines()
    block_replacements = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # FULLY PROTECT: Code blocks
        if token.type == "fence":
            placeholder = get_next_placeholder(token.content.strip())
            start, end = token.map
            block_replacements.append((start, end, f"```\n{placeholder}\n```"))

        # PARTIALLY PROTECT: Headers
        elif token.type == "heading_open":
            level = int(token.tag[1])
            inline_token = tokens[i + 1]
            header_text = inline_token.content.strip()
            placeholder = get_next_placeholder(header_text)
            start, end = token.map
            markdown_prefix = "#" * level
            block_replacements.append((start, end, f"{markdown_prefix} {placeholder}"))
            i += 2
            continue

        # HTML headers
        elif token.type == "html_block":
            html = token.content
            soup = BeautifulSoup(html, "html.parser")
            header_found = False
            for level in range(1, 7):
                tag = soup.find(f"h{level}")
                if tag:
                    header_text = tag.get_text(strip=True)
                    placeholder = get_next_placeholder(header_text)
                    start, end = token.map if token.map else (i, i + 1)
                    block_replacements.append((start, end, f"<h{level}>{placeholder}</h{level}>"))
                    header_found = True
                    break
            if not header_found and token.map:
                start, end = token.map
                block_replacements.append((start, end, html.strip()))

        # PARTIALLY PROTECT: Blockquotes
        elif token.type == "blockquote_open":
            start = token.map[0] if token.map else i
            j = i + 1
            blockquote_content = []
            while j < len(tokens) and tokens[j].type != "blockquote_close":
                if tokens[j].type == "paragraph_open":
                    para_token = tokens[j + 1]
                    if para_token.type == "inline":
                        blockquote_content.append(para_token.content)
                j += 1
            end = tokens[j].map[1] if tokens[j].map else start + 1
            if blockquote_content:
                text_content = " ".join(blockquote_content)
                placeholder = get_next_placeholder(text_content)
                block_replacements.append((start, end, f"> {placeholder}"))
            i = j

        i += 1

    # Apply block replacements
    for start, end, replacement in sorted(block_replacements, reverse=True):
        lines[start:end] = [replacement]

    # Inline protections
    processed_lines = []
    for line in lines:
        def replace_inline_code(match):
            return f"`{get_next_placeholder(match.group(1))}`"
        line = re.sub(r'`([^`]+)`', replace_inline_code, line)

        def replace_urls(match):
            return get_next_placeholder(match.group(0))
        line = re.sub(r'https?://[^\s\)\]\}]+', replace_urls, line)

        def replace_md_links(match):
            link_text = match.group(1)
            url = match.group(2)
            # Prevent wrapping already-wrapped placeholders
            if re.match(r'^<<__PLACEHOLDER_\d+__>>$', url):
                return match.group(0)
            return f"[{link_text}]({get_next_placeholder(url)})"
        line = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', replace_md_links, line)

        def replace_internal_links(match):
            link_text = match.group(1)
            anchor_target = f"#{match.group(2)}"
            if re.match(r'^<<__PLACEHOLDER_\d+__>>$', anchor_target):
                return match.group(0)
            return f"[{link_text}]({get_next_placeholder(anchor_target)})"
        line = re.sub(r'\[([^\]]+)\]\(#([^\)]+)\)', replace_internal_links, line)

        processed_lines.append(line)

    # Protect list bullets
    bullet_placeholder_lines = []
    for line in processed_lines:
        bullet_match = re.match(r'^(\s*)([-*+]|\d+\.)\s+', line)
        if bullet_match:
            indent, bullet = bullet_match.group(1), bullet_match.group(2)
            placeholder = get_next_placeholder(bullet, prefix="LIST_BULLET")
            line = re.sub(r'^(\s*)([-*+]|\d+\.)\s+', f"{indent}{placeholder} ", line)
        bullet_placeholder_lines.append(line)

    processed_lines = bullet_placeholder_lines
    
    raw_processed = "\n".join(processed_lines)

    # Replace horizontal rules
    raw_processed = re.sub(r'^\s*---\s*$', lambda m: get_next_placeholder(m.group(0)), raw_processed, flags=re.MULTILINE)

    # Replace newlines with unified placeholders
    final_lines = []
    for line in raw_processed.splitlines(keepends=True):
        newline_placeholder = get_next_placeholder("\n", prefix="PLACEHOLDER2")
        final_lines.append(line.rstrip('\n') + newline_placeholder)

    processed_content = ''.join(final_lines)

    # Prevent adjacent placeholder collisions
    processed_content = re.sub(
        r'(>>)(\s*<<)',  
        r'\1<<__PLACEHOLDER_SEPARATOR__>>\2',  
        processed_content
    )
    placeholder_map["__PLACEHOLDER_SEPARATOR__"] = ""  

    return placeholder_map, processed_content


def restore_placeholders(corrected_text: str, placeholder_map: Dict[str, str]) -> str:
    restored_text = corrected_text

    # Replace longer keys first to avoid prefix collisions
    for placeholder, original_content in sorted(placeholder_map.items(), key=lambda x: -len(x[0])):
        wrapped = f"<<{placeholder}>>"
        restored_text = restored_text.replace(wrapped, original_content)

    # Clean up separator
    restored_text = restored_text.replace('<<__PLACEHOLDER_SEPARATOR__>>', '')

    return restored_text

