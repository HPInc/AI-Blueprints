import base64
import logging
import os
from typing import Dict, Optional, Tuple, Union
from urllib.parse import urlparse

import requests

# Configure logger
logger: logging.Logger = logging.getLogger("run_workflow_logger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent duplicate logs from parent loggers

# Set formatter
formatter: logging.Formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Configure and attach stream handler
stream_handler: logging.StreamHandler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class GitHubMarkdownProcessor:
    """
    Processor for extracting and parsing Markdown files from GitHub repositories.

    This class fetches `.md` files from a public or private GitHub repository,
    replaces structural and inline components with labeled placeholders, and
    saves the resulting structure locally.

    Attributes:
        repo_url (str): GitHub repository URL.
        repo_owner (str): Owner of the GitHub repository.
        repo_name (str): Name of the GitHub repository.
        access_token (Optional[str]): GitHub Personal Access Token (if needed).
        save_dir (str): Directory to save the processed files.
        api_base_url (str): Base GitHub API URL for the repository.
    """

    def __init__(
        self,
        repo_url: str,
        access_token: Optional[str] = None,
        save_dir: str = "./parsed_repo",
    ):
        """
        Initializes the Markdown processor with a GitHub repo URL.

        Args:
            repo_url (str): Full URL to the GitHub repository.
            access_token (Optional[str]): GitHub token for private repo access.
            save_dir (str): Output directory to store processed Markdown files.
        """
        self.repo_url = repo_url
        self.access_token = access_token
        self.save_dir = save_dir

        owner, repo, error = self.parse_url()
        if error:
            raise ValueError(error)

        self.repo_owner = owner
        self.repo_name = repo
        self.api_base_url = (
            f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
        )

    def parse_url(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
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
        parsed_url = urlparse(self.repo_url)
        path_parts = parsed_url.path.strip("/").split("/")

        # Validate URL format
        if len(path_parts) < 2:
            return None, None, "Invalid GitHub URL format."

        # Return owner and repo
        return path_parts[0], path_parts[1], None

    def check_repo(self) -> str:
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
        owner, name, error = self.parse_url()
        if error:
            return error

        # Build GitHub URL
        url = f"https://api.github.com/repos/{owner}/{name}"

        # Build authentication header
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        response = requests.get(url, headers=headers)

        # Determine privacy of repo
        if response.status_code == 200:
            repo_data = response.json()
            return "private" if repo_data.get("private") else "public"
        elif response.status_code == 404:
            return "Repository is inaccessible. Please authenticate."
        else:
            return f"Error: {response.status_code}, {response.text}"

    def extract_md_files(self) -> Tuple[Optional[Dict], Optional[str]]:
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
        owner, name, error = self.parse_url()
        if error:
            return None, error

        # Build GitHub URL
        url = f"https://api.github.com/repos/{owner}/{name}"

        # Build authentication header
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

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
            path = item["path"]

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

        return dir_structure, None

    def run(self) -> Dict[str, str]:
        """
        High-level method to:
        1. Check repository access
        2. Extract markdown files
        3. Return raw markdown content by file path

        Returns:
            Dict[str, str]: Mapping from file paths to raw markdown content.
        """
        visibility = self.check_repo()
        logger.info(f"Repository visibility: {visibility}")

        if visibility == "Repository is inaccessible. Please authenticate.":
            raise PermissionError("Cannot access repository. Check your access token.")

        structure, error = self.extract_md_files()
        if error:
            raise RuntimeError(f"Markdown extraction failed: {error}")

        raw_data = {}

        def process_structure(
            structure: Dict[str, Union[str, dict]], path: str = ""
        ) -> None:
            """
            Recursively flattens a nested directory structure of markdown files.

            Args:
                structure (Dict[str, Union[str, dict]]): Nested dictionary representing directories and markdown file contents.
                path (str, optional): Current path used to build the full file path during traversal. Defaults to "".

            Returns:
                None: Updates the outer `raw_data` dictionary in-place with path-to-content mappings.
            """
            for name, content in structure.items():
                current_path = os.path.join(path, name)
                if isinstance(content, dict):
                    process_structure(content, current_path)
                else:
                    raw_data[current_path] = content

        process_structure(structure)
        logger.info("Raw markdown extraction complete.")
        return raw_data
