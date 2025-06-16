import requests
from urllib.parse import urlparse
import sys
import base64 
import re 

import os

# Input URL parser to extract owner and repo
def parse_url(github_url):
    parsed_url = urlparse(github_url)
    path_parts = parsed_url.path.strip("/").split("/")

    # Validate URL format
    if len(path_parts) < 2:
        return None, None, "Invalid GitHub URL format."
    
    # Return owner and repo
    return path_parts[0], path_parts[1], None

# Repo access checker
def check_repo(github_url, access_token=None):
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
def extract_md_files(github_url, access_token=None):
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
def parse_md(md_content):
    # Dictionary to store special components
    results = {
        "headers": {},
        "urls": {},
        "code_blocks": {},
        "inline_code": {},
    }

    # Copy of md content to be edited with placeholders
    placeholder_md = md_content

    # Dictionary to keep track of component counts for identification purposes
    counts = {
        "header": 0,
        "subheader": 0,
        "subsubheader": 0,
        "url": 0,
        "inline_code": 0,
        "code_block": 0,
    }

    # Code block replacer
    def replace_code_blocks(content):
        # Custom handler for re.sub()
        def repl(match):
            # Increment code block count, generate label, and store
            counts["code_block"] += 1
            label = f"CODEBLOCK_{counts['code_block']}"
            code = match.group(0).strip("`\n")
            results["code_blocks"][label] = code
            return label
        # Apply regex substitution
        return re.sub(r'```[\s\S]*?```', repl, content)

    # Inline code replacer
    def replace_inline_code(content):
        # Custom handler for re.sub()
        def repl(match):
            # Increment inline code count, generate label, and store
            counts["inline_code"] += 1
            label = f"INLINE_CODE_{counts['inline_code']}"
            code = match.group(1)
            results["inline_code"][label] = code
            return label
        # Apply regex substitution
        return re.sub(r'`([^`\n]+)`', repl, content)
    
    # Header replacer
    def replace_headers(content):
        # Custom handler for re.sub()
        def header_repl(match):
            # Interpret md headings
            hashes, text = match.group(1), match.group(2).strip()
            level = len(hashes)
            if level == 1:
                key, prefix = "header", "HEADER"
            elif level == 2:
                key, prefix = "subheader", "SUBHEADER"
            else:
                key, prefix = "subsubheader", "SUBSUBHEADER"
            # Increment respective heading count, generate label, and store
            counts[key] += 1
            label = f"{prefix}_{counts[key]}"
            results["headers"][label] = text
            return label
        # Apply regex substitution
        return re.sub(r'^(#{1,6})\s+(.*)', header_repl, content, flags=re.MULTILINE)
    
    # URL replacer
    def replace_urls(content):
        # Custom handler for re.sub()
        def repl(match):
            # Increment URL count, generate label, and store
            counts["url"] += 1
            label = f"URL_{counts['url']}"
            url = match.group(0)
            results["urls"][label] = url
            return label
        # Apply regex substitution
        return re.sub(r'https?://[^\s\)\]\}]+', repl, content)
    
    # Apply md preprocessing functions
    placeholder_md = replace_code_blocks(placeholder_md)
    placeholder_md = replace_inline_code(placeholder_md)
    placeholder_md = replace_headers(placeholder_md)
    placeholder_md = replace_urls(placeholder_md)

    return results, placeholder_md

# Repo tree traverser and parse caller
def traverse_and_parse(structure):
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
def save_files(structure, output_dir="."):
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

# Top level workflow
def main():
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
    
    # Initialize lists to store processed md content and extracted special elements
    processed_md = []
    extracted_elements = []

    # Traverse md file tree to parse and store content
    parsed_structure = traverse_and_parse(dir_structure)

    # Save files
    save_files(parsed_structure, output_dir="parsed_md_output")

    # Print content and extracted elements
    #print(processed_md)
    #print(extracted_elements)

if __name__ == "__main__":
    main()