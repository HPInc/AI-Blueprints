import requests
from urllib.parse import urlparse
import sys
import base64 
import re 
from markdown_it import MarkdownIt
from markdown_it.token import Token 
import os
from bs4 import BeautifulSoup
from typing import Optional, Tuple, Dict, Union

# Input URL parser to extract owner and repo
def parse_url(github_url: str) -> Tuple[Optional[str], Optional[str], Optional[str]] :
    """
    Parses a GitHub URL and extracts the repository owner and name.

    Args:
        github_url (str): The full GitHub URL to parse.

    Returns:
        Tuple[Optional[str], Optional[str], Optional[str]]:
            - The repository owner (e.g., 'openai')
            - The repository name (e.g., 'whisper')
            - An error message string if parsing fails; otherwise, None
    """
    parsed_url = urlparse(github_url)
    path_parts = parsed_url.path.strip("/").split("/")

    # Validate URL format
    if len(path_parts) < 2:
        return None, None, "Invalid GitHub URL format."
    
    # Return owner and repo
    return path_parts[0], path_parts[1], None

# Repo access checker
def check_repo(github_url: str, access_token: Optional[str] = None) -> str:
    """
    Determines the visibility (public or private) of a GitHub repository.

    Parses the provided GitHub URL, queries the GitHub API to fetch repository details,
    and returns a string indicating its visibility or an error message if access fails.

    Args:
        github_url (str): The URL of the GitHub repository to check.
        access_token (Optional[str], optional): GitHub personal access token for authenticated requests. Defaults to None.

    Returns:
        str: "private" or "public" if the repository is accessible;
             otherwise, a descriptive error message.
    """
    # Parse url into components
    owner, name, error = parse_url(github_url)
    if error:
        return error
    
    # Build GitHub URL
    url = f"https://api.github.com/repos/{owner}/{name}"

    # Build authentication header
    headers = {}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    response = requests.get(url, headers=headers)

    # Determine privacy of repo
    if response.status_code == 200:
        repo_data = response.json()
        return "private" if repo_data.get("private") else "public"
    elif response.status_code == 404:
        return "Repository is inaccessible. Please authenticate."
    else:
        return f"Error: {response.status_code}, {response.text}"
    
# Repo traverser to extract md files and save repo structure
def extract_md_files(github_url: str, access_token: Optional[str] = None) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Traverses a GitHub repository to extract all Markdown (.md) files and organize them in a nested directory structure.

    Connects to the GitHub API, retrieves the file tree of the default branch, downloads any Markdown files,
    and reconstructs their paths locally in a dictionary format. Supports optional authentication with a personal access token.

    Args:
        github_url (str): The GitHub repository URL (e.g., "https://github.com/user/repo").
        access_token (Optional[str], optional): GitHub personal access token for authenticated requests. Defaults to None.

    Returns:
        Tuple[Optional[Dict], Optional[str]]:
            - A nested dictionary representing the directory structure and Markdown file contents.
            - An error message string if any step fails; otherwise, None.
    """
    # Parse url into components
    owner, name, error = parse_url(github_url)
    if error:
        return None, error
    
    # Build GitHub URL
    url = f"https://api.github.com/repos/{owner}/{name}"

    # Build authentication header
    headers = {}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    response = requests.get(url, headers=headers)
    if not response.ok:
        return None, f"Error: {response.status_code}, {response.text}"

    # Get default branch
    default_branch = response.json().get("default_branch", "main")

    # Build GitHub URL for file tree
    tree_url = f"https://api.github.com/repos/{owner}/{name}/git/trees/{default_branch}?recursive=1"
    tree_response = requests.get(tree_url, headers=headers)
    if not tree_response.ok:
        return None, f"Error: {tree_response.status_code}, {tree_response.text}"
    
    # Dictionary to hold directory structure
    dir_structure = {}

    # Iterate through repo tree structure
    for item in tree_response.json().get("tree", []):
        path=item["path"]
    
        # Skip all non md files
        if item["type"] != "blob" or not path.endswith(".md"):
            continue

        # Fetch md file content
        content_url = f"https://api.github.com/repos/{owner}/{name}/contents/{path}"
        content_response = requests.get(content_url, headers=headers)
        if not content_response.ok:
            continue

        # Decode content response
        file_data = content_response.json()
        try:
            content = base64.b64decode(file_data["content"]).decode("utf-8")
        except Exception as e:
            content = f"Error decoding content: {e}"

        # Build directory structure
        parts = path.split("/")
        current = dir_structure 
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = content 

    return dir_structure

# Markdown parser to extract and placehold special text
def parse_md(md_content: str) -> Tuple[Dict[str, Dict[str, str]], str]:
    """
    Parses Markdown content to extract and label structural and inline elements.

    This function identifies and extracts:
        - Headers (Markdown and HTML-based)
        - Fenced code blocks
        - Blockquotes
        - Inline code
        - URLs
        - Internal markdown links

    Each identified element is replaced with a unique placeholder label (e.g., HEADER_1, CODEBLOCK_2),
    and the original content is stored in a structured dictionary for reference.

    Args:
        md_content (str): The raw Markdown content to parse.

    Returns:
        Tuple[Dict[str, Dict[str, str]], str]:
            - A dictionary containing extracted elements grouped by type and labeled sequentially.
            - A modified Markdown string with placeholders replacing the original elements.
    """
    # Initialize parser and parse Markdown into tokens
    md = MarkdownIt()
    tokens = md.parse(md_content)

    # Dictionary to store special components
    results = {
        "headers": {},
        "urls": {},
        "code_blocks": {},
        "inline_code": {},
        "blockquotes": {},
        "internal_links": {},
    }

    # Dictionary to keep track of component counts for identification purposes
    counts = {
        "header": 0,
        "subheader": 0,
        "subsubheader": 0,
        "url": 0,
        "inline_code": 0,
        "code_block": 0,
        "blockquote": 0,
        "internal_link": 0,
    }

    # Split md into lines
    lines = md_content.splitlines()

    # List to store line number ranges for clean replacements
    block_replacements = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Handle fenced code blocks
        if token.type == "fence":
            counts["code_block"] += 1
            label = f"CODEBLOCK_{counts['code_block']}"
            results["code_blocks"][label] = token.content.strip()
            start, end = token.map
            block_replacements.append((start, end, label))

        # Handle markdown headings
        elif token.type == "heading_open":
            level = int(token.tag[1])
            inline_token = tokens[i + 1]
            text = inline_token.content.strip()

            if level == 1:
                key, prefix = "header", "HEADER"
            elif level == 2:
                key, prefix = "subheader", "SUBHEADER"
            else:
                key, prefix = "subsubheader", "SUBSUBHEADER"

            counts[key] += 1
            label = f"{prefix}_{counts[key]}"
            results["headers"][label] = text
            start, end = token.map
            block_replacements.append((start, end, label))
            i += 2
            continue

        # Handle HTML blocks
        elif token.type == "html_block":
            html = token.content
            soup = BeautifulSoup(html, "html.parser")

            header_found = False
            for level in range(1, 7):
                tag = soup.find(f"h{level}")
                if tag:
                    text = tag.get_text(strip=True)
                    if level == 1:
                        key, prefix = "header", "HEADER"
                    elif level == 2:
                        key, prefix = "subheader", "SUBHEADER"
                    else:
                        key, prefix = "subsubheader", "SUBSUBHEADER"

                    counts[key] += 1
                    label = f"{prefix}_{counts[key]}"
                    results["headers"][label] = text
                    if token.map:
                        start, end = token.map
                        block_replacements.append((start, end, label))
                    else:
                        # fallback: insert at line after previous known block
                        start = end = i
                        block_replacements.append((start, start + 1, label))
                    header_found = True
                    break

            if not header_found and token.map:
                start, end = token.map
                block_replacements.append((start, end, html.strip()))
    
        # Handle blockquotes
        elif token.type == "blockquote_open":
            start = token.map[0] if token.map else i
            j = i + 1
            while j < len(tokens) and tokens[j].type != "blockquote_close":
                j += 1

            end = tokens[j].map[1] if tokens[j].map else start + 1
            blockquote_text = "\n".join(lines[start:end]).strip()
            counts["blockquote"] += 1
            label = f"BLOCKQUOTE_{counts['blockquote']}"
            results["blockquotes"][label] = blockquote_text
            block_replacements.append((start, end, label))
            i = j  # skip to close token
        i += 1

    # Replace all block elements in reverse to preserve line indexing
    for start, end, label in sorted(block_replacements, reverse=True):
        lines[start:end] = [label]

    # Define and apply patterns for inline replacements
    url_pattern = re.compile(r'https?://[^\s\)\]\}]+')
    inline_code_pattern = re.compile(r'`([^`]+)`')
    internal_link_pattern = re.compile(r'\[([^\]]+)\]\(#([^\)]+)\)')

    def replace_inline(line):
        # Handle internal markdown links
        def internal_link_repl(m):
            counts["internal_link"] += 1
            label_text = m.group(1)
            anchor = m.group(2)
            label = f"INTERNAL_LINK_{counts['internal_link']}"
            results["internal_links"][label] = {
                "text": label_text,
                "anchor": anchor,
            }
            return label

        line = internal_link_pattern.sub(internal_link_repl, line)

        # Replace inline code
        def inline_code_repl(m):
            counts["inline_code"] += 1
            label = f"INLINE_CODE_{counts['inline_code']}"
            results["inline_code"][label] = m.group(1)
            return label

        line = inline_code_pattern.sub(inline_code_repl, line)

        # Replace URLs
        def url_repl(m):
            counts["url"] += 1
            label = f"URL_{counts['url']}"
            results["urls"][label] = m.group(0)
            return label

        return url_pattern.sub(url_repl, line)

    # Apply inline replacements line by line (skip placeholder labels)
    for idx, line in enumerate(lines):
        if re.match(r'^(CODEBLOCK|HEADER|SUBHEADER|SUBSUBHEADER|BLOCKQUOTE)_\d+$', line.strip()):
            continue
        lines[idx] = replace_inline(line)

    # Return parsed results and placeholder-filled markdown
    placeholder_md = "\n".join(lines)
    return results, placeholder_md

# Repo tree traverser and parse caller
def traverse_and_parse(structure: Dict[str, Union[str, dict]]) -> Dict[str, Union[str, dict]]:
    """
    Recursively traverses a nested dictionary of Markdown files and replaces their content with placeholder-labeled versions.

    For each markdown string value, the function applies `parse_md()` to extract and replace structural elements with labeled placeholders. 
    Nested dictionaries are processed recursively to preserve the original hierarchy.

    Args:
        structure (Dict[str, Union[str, dict]]): A nested dictionary representing directory structure and Markdown file contents.

    Returns:
        Dict[str, Union[str, dict]]: A new dictionary with the same structure, where Markdown strings are replaced
        by their placeholder-filled versions.
    """
    parsed = {}
    for key, value in structure.items():
        if isinstance(value, dict):
            # Recurse into subdirectory
            parsed[key] = traverse_and_parse(value)
        elif isinstance(value, str):
            # Parse and replace markdown content
            _, placeholder = parse_md(value) 
            parsed[key] = placeholder
    return parsed

# Save processed files in a directory that emulates the structure of the repo
def save_files(structure: Dict[str, Union[str, dict]], output_dir: str =".") -> None:
    """
    Recursively saves a nested dictionary structure to disk, preserving the directory hierarchy.

    Each string value is written to a file, and each nested dictionary is treated as a subdirectory.
    This function mirrors the structure of a parsed repository and writes its contents to the specified output directory.

    Args:
        structure (Dict[str, Union[str, dict]]): A nested dictionary representing file paths and their contents.
        output_dir (str, optional): The root directory where the structure will be saved. Defaults to the current directory.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)  

    # Loop through every item in the structure dictionary
    for key, value in structure.items():
        # If value is a dictionary, it is a subfolder
        if isinstance(value, dict):
            new_dir = os.path.join(output_dir, key)
            os.makedirs(new_dir, exist_ok=True)
            save_files(value, new_dir)
        # If value is a string, it is a file
        elif isinstance(value, str):
            file_path = os.path.join(output_dir, key)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(value)

# Top level workflow (Will be removed)
def main() -> None:
    """
    Temporary CLI entry point for development and testing.
    """
    # Placeholder repo for testing purposes
    github_url = "https://github.com/HPInc/AI-Blueprints/tree/main"

    # Check repo visibility without authentication
    access_token = None
    visibility = check_repo(github_url, access_token)
    print(f"Visibility: {visibility}")

    # Prompt for auth token if inaccessible
    if visibility == "Repository is inaccessible. Please authenticate.":
        access_token = input("Enter your GitHub access token: ")

        # Check again with authentication
        status = check_repo(github_url, access_token)
        if status == "Repository is inaccessible. Please authenticate.":
            print("Unable to locate/access repository.")
            sys.exit(0)
        else:
            print(f"Visibility: {status}")

    # Obtain md files along with directory structure
    dir_structure = extract_md_files(github_url, access_token)
    
    # Traverse md file tree to parse and store content
    parsed_structure = traverse_and_parse(dir_structure)

    # Save files
    save_files(parsed_structure, output_dir="parsed_md_output")

if __name__ == "__main__":
    main()