import re
from typing import Dict, Tuple
from markdown_it import MarkdownIt


def parse_md_for_grammar_correction(md_content: str) -> Tuple[Dict[str, str], str]:
    """
    Parses Markdown content by replacing non-prose elements with placeholders.

    Args:
        md_content (str): Raw markdown content to process.

    Returns:
        Tuple[Dict[str, str], str]:
            - A dictionary mapping placeholder keys to original markdown blocks.
            - The transformed markdown with placeholders in place of structure.
    """

    md = MarkdownIt()

    placeholder_map = {}
    counter = 0

    def get_next_placeholder(value: str, prefix="PH") -> str:
        """
        Generates a unique placeholder token for the given value and stores the mapping.

        If the input contains any previously assigned placeholders, they are unwrapped and reassigned
        as a new single placeholder. This helps deduplicate and simplify complex inline structures.

        Args:
            value (str): The text content to be replaced by a placeholder.
            prefix (str): The placeholder prefix (e.g., "PH", "BULLET").

        Returns:
            str: The generated placeholder token (e.g., "<<PH3>>" or "[[BULLET2]]").
        """
        nonlocal counter
        value = re.sub(
            r"<<(PH|BULLET|SEP)\d*>>|\[\[BULLET\d+\]\]",
            lambda m: placeholder_map.get(m.group(0).strip("<>[]"), m.group(0)),
            value,
        )
        counter += 1
        key = f"{prefix}{counter}"
        placeholder_map[key] = value

        if prefix == "BULLET":
            return f"[[{key}]]"
        else:
            return f"<<{key}>>"

    def protect_front_matter(content: str) -> str:
        """
        Detects and replaces a YAML front matter block with a single placeholder.
        The front matter must be at the very beginning of the string.

        Args:
            content (str): Input markdown content.

        Returns:
            str: Markdown with front matter replaced by a placeholder.
        """
        front_matter_pattern = re.compile(r"\A---\s*\n.*?\n---\s*\n?", re.DOTALL)

        match = front_matter_pattern.search(content)
        if match:
            front_matter_block = match.group(0)
            placeholder = get_next_placeholder(front_matter_block)
            return front_matter_pattern.sub(placeholder, content, count=1)

        return content

    def protect_tables(content: str) -> str:
        """
        Detects Markdown tables and replaces them with placeholders.

        Args:
            content (str): Input markdown content.

        Returns:
            str: Markdown with tables replaced by placeholders.
        """
        lines = content.splitlines()
        protected_lines = []
        in_table = False
        table_buffer = []

        for line in lines:
            if (
                "|" in line
                and line.strip().startswith("|")
                and line.strip().endswith("|")
            ):
                if not in_table:
                    in_table = True
                    table_buffer = [line]
                else:
                    table_buffer.append(line)
            elif in_table and re.match(r"^\s*\|[\s\-\|:]+\|\s*$", line):
                table_buffer.append(line)
            elif in_table:
                if len(table_buffer) >= 2:
                    table_content = "\n".join(table_buffer)
                    placeholder = get_next_placeholder(table_content)
                    protected_lines.append(placeholder)
                else:
                    protected_lines.extend(table_buffer)

                table_buffer = []
                in_table = False
                protected_lines.append(line)
            else:
                protected_lines.append(line)

        # Handle final table
        if in_table and len(table_buffer) >= 2:
            table_content = "\n".join(table_buffer)
            placeholder = get_next_placeholder(table_content)
            protected_lines.append(placeholder)
        elif in_table:
            protected_lines.extend(table_buffer)

        return "\n".join(protected_lines)

    def is_low_prose_line(line: str, threshold: float = 0.1) -> bool:
        """
        Determines if a line contains mostly structure and little natural language prose.

        Args:
            line (str): A single markdown line.
            threshold (float): Minimum proportion of prose content.

        Returns:
            bool: True if it's mostly non-prose.
        """
        if not line.strip():
            return True
        no_placeholders = re.sub(r"<<(PH|BULLET|SEP)\d*>>|\[\[BULLET\d+\]\]", "", line)
        no_markdown = re.sub(r"[*_`[\]()#|-]", "", no_placeholders)
        prose_content = no_markdown.strip()
        if len(line) > 0 and (len(prose_content) / len(line)) < threshold:
            return True
        return False

    md_content = protect_front_matter(md_content)
    md_content = protect_tables(md_content)

    tokens = md.parse(md_content)
    lines = md_content.splitlines()
    block_replacements = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if (
            token.type == "html_block"
            and i + 3 < len(tokens)
            and tokens[i + 1].type == "heading_open"
        ):
            html_start, _ = token.map
            _, heading_end = tokens[i + 1].map
            raw_block = "\n".join(lines[html_start:heading_end])
            placeholder = get_next_placeholder(raw_block)
            block_replacements.append((html_start, heading_end, placeholder))
            i += 4
            continue
        elif token.type == "fence" or token.type == "html_block":
            start, end = token.map
            raw_block = "\n".join(lines[start:end])
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
                if (
                    tokens[j].type == "paragraph_open"
                    and j + 1 < len(tokens)
                    and tokens[j + 1].type == "inline"
                ):
                    blockquote_content.append(tokens[j + 1].content)
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

    def replace_md_links(match):
        """
        Replaces standard Markdown links with placeholders for the URL target.
        """
        text, url = match.group(1), match.group(2)
        if re.match(r"^<<PH\d+>>$", url):
            return match.group(0)
        return f"[{text}]({get_next_placeholder(url)})"

    def replace_internal_links(match):
        """
        Replaces internal anchor links with placeholders for the anchor target.
        """
        text, anchor = match.group(1), f"#{match.group(2)}"
        if re.match(r"^<<PH\d+>>$", anchor):
            return match.group(0)
        return f"[{text}]({get_next_placeholder(anchor)})"

    processed_lines = []
    for line in lines:
        # Protect inline code first, as it can contain any character.
        line = re.sub(
            r"`([^`]+)`", lambda m: f"`{get_next_placeholder(m.group(1))}`", line
        )

        # Protect Markdown images
        line = re.sub(
            r"!\[([^\]]*)\]\(([^)]+)\)",
            lambda m: get_next_placeholder(m.group(0)),
            line,
        )

        # Protect standard Markdown links []() and internal links [](#).
        line = re.sub(r"\[([^\]]+)]\(([^)]+)\)", replace_md_links, line)
        line = re.sub(r"\[([^\]]+)]\(#([^)]+)\)", replace_internal_links, line)

        # Protect raw URLs last, as they are the most generic.
        line = re.sub(
            r"https?://[^\s)\]}]+", lambda m: get_next_placeholder(m.group(0)), line
        )

        processed_lines.append(line)

    def is_title_line(content: str) -> bool:
        """Heuristic check to determine if a line is a title-style phrase."""
        clean = re.sub(r"<<[^>]+>>", "", content)
        clean = re.sub(r"[*_`[\]\(\)]", "", clean).strip()
        words = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", clean)
        if len(words) < 2:
            return False
        alpha = re.sub(r"[^A-Za-z]", "", clean)
        if alpha.isupper():
            return True
        upper_words = [w for w in words if w[0].isupper()]
        if len(words) > 0 and len(upper_words) / len(words) >= 0.75:
            return True
        return False

    bullet_placeholder_lines = []
    for line in processed_lines:
        if is_low_prose_line(line):
            placeholder = get_next_placeholder(line)
            bullet_placeholder_lines.append(placeholder)
            continue

        m = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", line)
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
    raw_processed = re.sub(
        r"^\s*---\s*$",
        lambda m: get_next_placeholder(m.group(0)),
        raw_processed,
        flags=re.MULTILINE,
    )

    final_lines = []
    for line in raw_processed.splitlines(keepends=True):
        if line.endswith("\n"):
            content = line.rstrip("\n")
            trailing_newline = True
        else:
            content = line
            trailing_newline = False
        newline_placeholder = get_next_placeholder("\n", prefix="PH")
        final_lines.append(content + (newline_placeholder if trailing_newline else ""))

    processed_content = "".join(final_lines)

    merged_placeholder_map = {}
    pattern = re.compile(r"(?:<<PH\d+>>){2,}")

    while True:
        match = pattern.search(processed_content)
        if not match:
            break
        ph_sequence = re.findall(r"<<PH\d+>>", match.group(0))
        keys = [ph.strip("<>") for ph in ph_sequence]
        merged_value = "".join(placeholder_map.get(k, "") for k in keys)
        counter += 1
        new_key = f"PH{counter}"
        new_ph = f"<<{new_key}>>"
        placeholder_map[new_key] = merged_value
        processed_content = (
            processed_content[: match.start()]
            + new_ph
            + processed_content[match.end() :]
        )
        merged_placeholder_map[new_key] = keys

    processed_content = re.sub(
        r"(<<PH\d+>>)(?!<<)(?=\w)", r"\1<<SEP>>", processed_content
    )
    placeholder_map["SEP"] = ""

    return placeholder_map, processed_content


def restore_placeholders(corrected_text: str, placeholder_map: Dict[str, str]) -> str:
    """
    Replaces placeholders in the corrected markdown content with their original values.

    Args:
        corrected_text (str): Markdown content containing placeholders.
        placeholder_map (Dict[str, str]): Map of placeholders to original content.

    Returns:
        str: Fully restored markdown with original formatting.
    """
    restored_text = corrected_text

    # Replace placeholder tokens with original values
    # Sort by length descending to handle nested placeholders correctly
    for placeholder, original in sorted(
        placeholder_map.items(), key=lambda x: -len(x[0])
    ):
        if placeholder.startswith("BULLET"):
            restored_text = restored_text.replace(f"[[{placeholder}]]", original)
        else:
            restored_text = restored_text.replace(f"<<{placeholder}>>", original)

    # Remove SEP markers
    restored_text = restored_text.replace("<<SEP>>", "")

    return restored_text
