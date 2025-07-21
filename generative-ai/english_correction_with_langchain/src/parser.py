from typing import Tuple, Dict
import re
from markdown_it import MarkdownIt

def parse_md_for_grammar_correction(md_content: str) -> Tuple[Dict[str, str], str]:
    """
    Parses Markdown content by replacing non-prose elements with placeholders.
    This version uses a simplified, robust method for HTML blocks while keeping
    all other original logic intact, including newline placeholders.
    """

    md = MarkdownIt()
        
    placeholder_map = {}
    counter = 0

    def get_next_placeholder(value: str, prefix="PH") -> str:
        nonlocal counter
        value = re.sub(
            r'<<(PH|BULLET|SEP)\d*>>|\[\[BULLET\d+\]\]',
            lambda m: placeholder_map.get(m.group(0).strip('<>[]'), m.group(0)),
            value
        )
        counter += 1
        key = f"{prefix}{counter}"
        placeholder_map[key] = value
        
        if prefix == "BULLET":
            return f"[[{key}]]"
        else:
            return f"<<{key}>>"
    
    def protect_tables(content: str) -> str:
        lines = content.splitlines()
        protected_lines = []
        in_table = False
        table_buffer = []
        
        for line in lines:
            if '|' in line and line.strip().startswith('|') and line.strip().endswith('|'):
                if not in_table:
                    in_table = True
                    table_buffer = [line]
                else:
                    table_buffer.append(line)
            elif in_table and re.match(r'^\s*\|[\s\-\|:]+\|\s*$', line):
                table_buffer.append(line)
            elif in_table:
                if len(table_buffer) >= 2:
                    table_content = '\n'.join(table_buffer)
                    placeholder = get_next_placeholder(table_content)
                    protected_lines.append(placeholder)
                else:
                    protected_lines.extend(table_buffer)
                
                table_buffer = []
                in_table = False
                protected_lines.append(line)
            else:
                protected_lines.append(line)
        
        if in_table and len(table_buffer) >= 2:
            table_content = '\n'.join(table_buffer)
            placeholder = get_next_placeholder(table_content)
            protected_lines.append(placeholder)
        elif in_table:
            protected_lines.extend(table_buffer)
        
        return '\n'.join(protected_lines)

    def is_low_prose_line(line: str, threshold: float = 0.1) -> bool:
        """
        Determines if a line consists mostly of placeholders and syntax.
        """
        if not line.strip():
            return True # Treat empty lines as low prose

        # Create a "clean" version by removing all known non-prose elements
        no_placeholders = re.sub(r'<<(PH|BULLET|SEP)\d*>>|\[\[BULLET\d+\]\]', '', line)
        
        # Remove common markdown characters
        no_markdown = re.sub(r'[*_`[\]()#|-]', '', no_placeholders)
        
        # What's left is considered "prose"
        prose_content = no_markdown.strip()
        
        # If the ratio of prose to the total line length is below the threshold, protect it
        if len(line) > 0 and (len(prose_content) / len(line)) < threshold:
            return True
            
        return False
    
    md_content = protect_tables(md_content)
    
    tokens = md.parse(md_content)
    lines = md_content.splitlines()
    block_replacements = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # First, check for the special case of an HTML block followed by a heading.
        if (token.type == "html_block" 
            and i + 3 < len(tokens) 
            and tokens[i + 1].type == "heading_open"):
            
            html_start, _ = token.map
            _, heading_end = tokens[i + 1].map 

            raw_block = '\n'.join(lines[html_start:heading_end])
            placeholder = get_next_placeholder(raw_block)
            block_replacements.append((html_start, heading_end, placeholder))
            
            i += 4 
            continue

        elif token.type == "fence" or token.type == "html_block":
            start, end = token.map
            raw_block = '\n'.join(lines[start:end])
            placeholder = get_next_placeholder(raw_block)
            block_replacements.append((start, end, placeholder))
            i += 1
            continue
            
        elif token.type == "heading_open":
            level = int(token.tag[1])
            inline_token = tokens[i + 1]
            header_text = inline_token.content.strip()
            placeholder = get_next_placeholder(header_text)
            start, end = token.map
            markdown_prefix = "#" * level
            block_replacements.append((start, end, f"{markdown_prefix} {placeholder}"))
            i += 3
            continue

        elif token.type == "blockquote_open":
            start = token.map[0] if token.map else i
            j = i + 1
            blockquote_content = []
            while j < len(tokens) and tokens[j].type != "blockquote_close":
                if tokens[j].type == "paragraph_open" and j + 1 < len(tokens) and tokens[j+1].type == "inline":
                    blockquote_content.append(tokens[j+1].content)
                j += 1
            
            if j < len(tokens):
                end = tokens[j].map[1] if tokens[j].map else start + 1
                if blockquote_content:
                    text_content = " ".join(blockquote_content)
                    placeholder = get_next_placeholder(text_content)
                    block_replacements.append((start, end, f"> {placeholder}"))
                i = j + 1
            else:
                i += 1
            continue
            
        i += 1

    for start, end, replacement in sorted(block_replacements, reverse=True):
        lines[start:end] = [replacement]

    # Define helper functions once, outside the loop
    def replace_md_links(match):
        text, url = match.group(1), match.group(2)
        if re.match(r'^<<PH\d+>>$', url):
            return match.group(0)
        return f"[{text}]({get_next_placeholder(url)})"

    def replace_internal_links(match):
        text, anchor = match.group(1), f"#{match.group(2)}"
        if re.match(r'^<<PH\d+>>$', anchor):
            return match.group(0)
        return f"[{text}]({get_next_placeholder(anchor)})"

    processed_lines = []
    for line in lines:
        line = re.sub(r'`([^`]+)`', lambda m: f"`{get_next_placeholder(m.group(1))}`", line)
        line = re.sub(r'https?://[^\s)\]}]+', lambda m: get_next_placeholder(m.group(0)), line)
        line = re.sub(r'\[([^\]]+)]\(([^)]+)\)', replace_md_links, line)
        line = re.sub(r'\[([^\]]+)]\(#([^)]+)\)', replace_internal_links, line)
        processed_lines.append(line)

    def is_title_line(content: str) -> bool:
        clean = re.sub(r'<<[^>]+>>', '', content)
        clean = re.sub(r'[*_`[\]\(\)]', '', clean).strip()
        words = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", clean)
        if len(words) < 2: return False
        alpha = re.sub(r'[^A-Za-z]', '', clean)
        if alpha.isupper(): return True
        upper_words = [w for w in words if w[0].isupper()]
        if len(upper_words) / len(words) >= 0.75: return True
        return False

    bullet_placeholder_lines = []
    for line in processed_lines:
        if is_low_prose_line(line):
            placeholder = get_next_placeholder(line)
            bullet_placeholder_lines.append(placeholder)
            continue

        m = re.match(r'^(\s*)([-*+]|\d+\.)\s+(.*)$', line)
        if m:
            indent, bullet, content = m.groups()
            if is_title_line(content):
                ph = get_next_placeholder(content)
                bullet_placeholder_lines.append(f"{indent}{bullet} {ph}")
            else:
                bph = get_next_placeholder(bullet, prefix="BULLET")
                bullet_placeholder_lines.append(f"{indent}{bph} {content}")
        else:
            bullet_placeholder_lines.append(line)

    raw_processed = "\n".join(bullet_placeholder_lines)
    raw_processed = re.sub(r'^\s*---\s*$', lambda m: get_next_placeholder(m.group(0)), raw_processed, flags=re.MULTILINE)

    final_lines = []
    for line in raw_processed.splitlines(keepends=True):
        if line.endswith('\n'):
            content = line.rstrip('\n')
            trailing_newline = True
        else:
            content = line
            trailing_newline = False
        
        newline_placeholder = get_next_placeholder("\n", prefix="PH")
        final_lines.append(content + (newline_placeholder if trailing_newline else ''))

    processed_content = ''.join(final_lines)

    merged_placeholder_map = {}
    pattern = re.compile(r'(?:<<PH\d+>>){2,}')
    
    while True:
        match = pattern.search(processed_content)
        if not match: break
    
        ph_sequence = re.findall(r'<<PH\d+>>', match.group(0))
        keys = [ph.strip('<>') for ph in ph_sequence]
        merged_value = ''.join(placeholder_map.get(k, '') for k in keys)
    
        counter += 1
        new_key = f"PH{counter}"
        new_ph = f"<<{new_key}>>"
        placeholder_map[new_key] = merged_value
    
        processed_content = processed_content[:match.start()] + new_ph + processed_content[match.end():]
        merged_placeholder_map[new_key] = keys

    processed_content = re.sub(r'(<<PH\d+>>)(?!<<)(?=\w)', r'\1<<SEP>>', processed_content)
    placeholder_map["SEP"] = ""
    
    return placeholder_map, processed_content


def restore_placeholders(corrected_text: str, placeholder_map: Dict[str, str]) -> str:
    """
    Restores the original content from the placeholders in the corrected text.
    """
    restored_text = corrected_text

    for placeholder, original in sorted(placeholder_map.items(), key=lambda x: -len(x[0])):
        if placeholder.startswith("BULLET"):
            restored_text = restored_text.replace(f"[[{placeholder}]]", original)
        else:
            restored_text = restored_text.replace(f"<<{placeholder}>>", original)

    restored_text = restored_text.replace('<<SEP>>', '')

    return restored_text