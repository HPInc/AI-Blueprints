from typing import Tuple, Dict

def parse_md_for_grammar_correction(md_content: str) -> Tuple[Dict[str, str], str]:
    """
    Parses Markdown content specifically for grammar correction tasks.
    
    - FULLY PROTECT: Code blocks, URLs, inline code (never need grammar correction)
    - PARTIALLY PROTECT: Headers, blockquotes (extract text for correction, preserve structure)
    - PRESERVE: Everything else for normal grammar correction
    
    Returns:
        Tuple[Dict[str, str], str]: (placeholder_map, processed_content)
    """
    import re
    from markdown_it import MarkdownIt
    from bs4 import BeautifulSoup
    
    md = MarkdownIt()
    tokens = md.parse(md_content)
    
    placeholder_map = {}
    counter = 0
    
    def get_next_placeholder():
        nonlocal counter
        counter += 1
        return f"__PLACEHOLDER_{counter}__"
    
    lines = md_content.splitlines()
    block_replacements = []
    
    # Process tokens for block-level elements
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # FULLY PROTECT: Code blocks (no grammar correction needed)
        if token.type == "fence":
            placeholder = get_next_placeholder()
            placeholder_map[placeholder] = token.content.strip()
            start, end = token.map
            block_replacements.append((start, end, f"```\n{placeholder}\n```"))
        
        # PARTIALLY PROTECT: Headers (extract text for correction)
        elif token.type == "heading_open":
            level = int(token.tag[1])
            inline_token = tokens[i + 1]
            header_text = inline_token.content.strip()
            
            # Create placeholder for correctable text
            placeholder = get_next_placeholder()
            placeholder_map[placeholder] = header_text
            
            # Replace with structure intact but text as placeholder
            start, end = token.map
            markdown_prefix = "#" * level
            block_replacements.append((start, end, f"{markdown_prefix} {placeholder}"))
            i += 2
            continue
            
        # Handle HTML headers similarly
        elif token.type == "html_block":
            html = token.content
            soup = BeautifulSoup(html, "html.parser")
            
            header_found = False
            for level in range(1, 7):
                tag = soup.find(f"h{level}")
                if tag:
                    header_text = tag.get_text(strip=True)
                    placeholder = get_next_placeholder()
                    placeholder_map[placeholder] = header_text
                    
                    start, end = token.map if token.map else (i, i+1)
                    block_replacements.append((start, end, f"<h{level}>{placeholder}</h{level}>"))
                    header_found = True
                    break
            
            # If not a header, preserve as-is (might be other HTML we want to keep)
            if not header_found and token.map:
                start, end = token.map
                block_replacements.append((start, end, html.strip()))
        
        # PARTIALLY PROTECT: Blockquotes (extract text for correction)
        elif token.type == "blockquote_open":
            start = token.map[0] if token.map else i
            j = i + 1
            
            # Find the blockquote content
            blockquote_content = []
            while j < len(tokens) and tokens[j].type != "blockquote_close":
                if tokens[j].type == "paragraph_open":
                    para_token = tokens[j + 1]
                    if para_token.type == "inline":
                        blockquote_content.append(para_token.content)
                j += 1
            
            end = tokens[j].map[1] if tokens[j].map else start + 1
            
            # Create placeholder for the correctable content
            if blockquote_content:
                text_content = " ".join(blockquote_content)
                placeholder = get_next_placeholder()
                placeholder_map[placeholder] = text_content
                block_replacements.append((start, end, f"> {placeholder}"))
            
            i = j
        
        i += 1
    
    # Apply block replacements in reverse order
    for start, end, replacement in sorted(block_replacements, reverse=True):
        lines[start:end] = [replacement]
    
    # Now handle inline elements
    processed_lines = []
    for line in lines:
        processed_line = line
        
        # FULLY PROTECT: Inline code
        def replace_inline_code(match):
            placeholder = get_next_placeholder()
            placeholder_map[placeholder] = match.group(1)
            return f"`{placeholder}`"
        
        processed_line = re.sub(r'`([^`]+)`', replace_inline_code, processed_line)
        
        # FULLY PROTECT: URLs (standalone)
        def replace_urls(match):
            placeholder = get_next_placeholder()
            placeholder_map[placeholder] = match.group(0)
            return placeholder
        
        processed_line = re.sub(r'https?://[^\s\)\]\}]+', replace_urls, processed_line)
        
        # FULLY PROTECT: Markdown links (preserve structure, protect URL)
        def replace_md_links(match):
            link_text = match.group(1)  # Keep text for grammar correction
            url = match.group(2)
            placeholder = get_next_placeholder()
            placeholder_map[placeholder] = url
            return f"[{link_text}]({placeholder})"
        
        processed_line = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', replace_md_links, processed_line)
        
        # FULLY PROTECT: Internal links
        def replace_internal_links(match):
            link_text = match.group(1)
            anchor = match.group(2)
            placeholder = get_next_placeholder()
            placeholder_map[placeholder] = f"#{anchor}"
            return f"[{link_text}]({placeholder})"
        
        processed_line = re.sub(r'\[([^\]]+)\]\(#([^\)]+)\)', replace_internal_links, processed_line)
        
        processed_lines.append(processed_line)
    
    processed_content = "\n".join(processed_lines)
    return placeholder_map, processed_content

def restore_placeholders(corrected_text: str, placeholder_map: Dict[str, str]) -> str:
    """
    Restores original content from placeholders after grammar correction.

    Args:
        corrected_text (str): The text containing placeholder tokens that need to be restored.
        placeholder_map (Dict[str, str]): A dictionary mapping placeholders to their original content.

    Returns:
        str: The text with all placeholders replaced by their original content.
    """
    restored_text = corrected_text
    
    # Replace placeholders with original content
    for placeholder, original_content in placeholder_map.items():
        restored_text = restored_text.replace(placeholder, original_content)
    
    return restored_text