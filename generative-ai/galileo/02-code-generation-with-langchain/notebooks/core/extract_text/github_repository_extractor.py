"""
GitHub Repository Extractor

Key Features:
- File size limits to prevent processing of large files that would exceed context windows
- Pattern-based filtering to exclude minified/bundled JavaScript and CSS files
- Intelligent code chunking for large files to stay within token limits
- Preserved context information with file path and line number references

Usage:
    extractor = GitHubRepositoryExtractor(
        repo_url="https://github.com/username/repo",
        max_file_size_kb=500,  # Limit files to 500KB
        max_chunk_size=100     # Break files into 100-line chunks
    )
    data = extractor.run()

"""

import os
import re
import requests
import shutil
import uuid
import logging
import nbformat
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional, Set, Tuple

class GitHubRepositoryExtractor:
    """
    Extractor for code and documentation from GitHub repositories.
    Works with multiple file types
    
    Attributes:
        repo_url (str): GitHub repository URL
        repo_owner (str): GitHub repository owner
        repo_name (str): GitHub repository name
        save_dir (str): Local directory to save downloaded files
        verbose (bool): If True, enables logging output
        max_file_size_kb (int): Maximum file size in KB to process
        max_chunk_size (int): Maximum size of code chunks (in lines)
        supported_extensions (tuple): File extensions this class will process
        excluded_patterns (set): Regex patterns for files to exclude
    """
    
    def __init__(self, repo_url: str, save_dir: str = './repo_files', verbose: bool = False,
                 max_file_size_kb: int = 500, max_chunk_size: int = 100,
                 supported_extensions: tuple = ('.py', '.ipynb', '.md', '.txt', '.json')):
        """
        Initializes the GitHub repository extractor.
        
        Args:
            repo_url (str): URL to the GitHub repository
            save_dir (str): Directory to save downloaded files
            verbose (bool): Whether to enable verbose logging
            max_file_size_kb (int): Maximum file size in KB to process
            max_chunk_size (int): Maximum size of code chunks in lines
            supported_extensions (tuple): File extensions to process
        """
        # Parse repository URL to extract owner and name
        parsed_url = self._parse_github_url(repo_url)
        self.repo_url = repo_url
        self.repo_owner = parsed_url["owner"]
        self.repo_name = parsed_url["repo"]
        self.save_dir = save_dir
        self.verbose = verbose
        self.max_file_size_kb = max_file_size_kb
        self.max_chunk_size = max_chunk_size
        self.supported_extensions = supported_extensions
        self.api_base_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/contents"
        
        # Define patterns for files to exclude (minified/bundled files that can exceed context)
        self.excluded_patterns = {
            r'\.min\.(js|css)$',  # Minified JS/CSS
            r'-[a-zA-Z0-9]{8}\.(js|css)$',  # Bundled files with hash
            r'bundle\.(js|css)$',  # Generic bundles
            r'vendor\.(js|css)$',  # Vendor bundles
            r'dist/assets/.*\.(js|css)$',  # Dist assets
            r'node_modules/.*',  # Node modules
        }
        # Define specific filenames to exclude (lock files, large auto-generated JSON)
        self.excluded_filenames = {
            'package-lock.json',
            'yarn.lock',
            'pnpm-lock.yaml',
            'package-lock.yaml'
        }
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.propagate = False
        self.logger.handlers.clear()
        
        if verbose:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.addHandler(logging.NullHandler())
    
    def _parse_github_url(self, url: str) -> Dict[str, str]:
        """
        Parses a GitHub URL to extract owner and repository name.
        
        Args:
            url (str): GitHub repository URL
            
        Returns:
            Dict with owner and repo names
            
        Raises:
            ValueError: If the URL is not a valid GitHub repository URL
        """
        # Parse the URL
        parsed = urlparse(url)
        
        # Validate that it's a GitHub URL
        if not parsed.netloc.endswith('github.com'):
            raise ValueError(f"Not a GitHub URL: {url}")
        
        # Extract path components
        path_parts = [p for p in parsed.path.split('/') if p]
        
        if len(path_parts) < 2:
            raise ValueError(f"Invalid GitHub repository URL: {url}")
            
        return {
            "owner": path_parts[0],
            "repo": path_parts[1]
        }
    
    def run(self) -> List[Dict]:
        """
        Main entry point - processes the repository and extracts code with context.
        
        Returns:
            List of dictionaries with extracted code and metadata
        """
        self.logger.info(f"Processing repository: {self.repo_owner}/{self.repo_name}")
        
        # Clean up save directory if it exists
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
            self.logger.info(f"Removed existing directory: {self.save_dir}")
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger.info(f"Created directory: {self.save_dir}")
        
        # Process the repository contents
        extracted_data = self._process_directory(self.api_base_url)
        self.logger.info(f"Extracted {len(extracted_data)} code snippets from repository")
        
        return extracted_data
    
    def _should_skip_file(self, file_path: str, size: int) -> bool:
        """
        Determines whether a file should be skipped based on size and pattern.
        
        Args:
            file_path (str): Path to the file
            size (int): Size of the file in bytes
            
        Returns:
            True if the file should be skipped, False otherwise
        """
        # Exclude specific lock or auto-generated JSON files by name
        basename = os.path.basename(file_path)
        if basename in getattr(self, 'excluded_filenames', {}):
            self.logger.info(f"Skipping excluded filename: {file_path}")
            return True
        # Check file size
        if size > self.max_file_size_kb * 1024:
            self.logger.info(f"Skipping large file ({size/1024:.1f} KB): {file_path}")
            return True
            
        # Check excluded patterns
        for pattern in self.excluded_patterns:
            if re.search(pattern, file_path):
                self.logger.info(f"Skipping excluded file pattern: {file_path}")
                return True
                
        return False
    
    def _process_directory(self, api_url: str) -> List[Dict]:
        """
        Recursively processes a directory in the repository.
        
        Args:
            api_url (str): GitHub API URL for the directory
            
        Returns:
            List of dictionaries with extracted code and metadata
        """
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            contents = response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch contents from {api_url}: {str(e)}")
            return []
        
        all_data = []
        
        for item in contents:
            try:
                if item['type'] == 'file':
                    file_extension = os.path.splitext(item['name'])[1].lower()
                    
                    # Only process files with supported extensions
                    if file_extension in self.supported_extensions:
                        # Skip files that are too large or match excluded patterns
                        if self._should_skip_file(item['path'], item['size']):
                            continue
                            
                        file_path = os.path.join(self.save_dir, item['path'])
                        
                        # Create directory structure if needed
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        
                        # Download and extract from the file
                        self._download_file(item['download_url'], file_path)
                        extracted = self._extract_from_file(file_path)
                        all_data.extend(extracted)
                
                elif item['type'] == 'dir':
                    # Process subdirectory recursively
                    subdir_api_url = item['url']
                    all_data.extend(self._process_directory(subdir_api_url))
            
            except Exception as e:
                self.logger.error(f"Error processing {item.get('path', 'unknown')}: {str(e)}")
        
        return all_data
    
    def _download_file(self, file_url: str, save_path: str) -> None:
        """
        Downloads a file from GitHub.
        
        Args:
            file_url (str): URL to download the file from
            save_path (str): Path to save the file to
        """
        try:
            response = requests.get(file_url)
            response.raise_for_status()
            
            # Create directory if needed
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"Downloaded: {save_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to download {file_url}: {str(e)}")
    
    def _extract_from_file(self, file_path: str) -> List[Dict]:
        """
        Extracts code and context from a file based on its type.
        
        Args:
            file_path (str): Path to the file to extract from
            
        Returns:
            List of dictionaries with extracted code and metadata
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.ipynb':
            return self._extract_from_notebook(file_path)
        else:
            return self._extract_from_code_file(file_path)
    
    def _extract_from_notebook(self, notebook_path: str) -> List[Dict]:
        """
        Extracts code cells and associated context from Jupyter notebooks.
        
        Args:
            notebook_path (str): Path to the notebook file
            
        Returns:
            List of dictionaries with extracted code and metadata
        """
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            extracted_data = []
            context = ''
            
            for cell in notebook['cells']:
                if cell['cell_type'] == 'markdown':
                    # Use markdown cells as context for subsequent code cells
                    context = ''.join(cell['source'])
                
                elif cell['cell_type'] == 'code' and cell['source'].strip():
                    # For large code cells, break into smaller chunks
                    code_chunks = self._chunk_code(''.join(cell['source']), self.max_chunk_size)
                    
                    for i, chunk in enumerate(code_chunks):
                        # Add chunk indicator if multiple chunks
                        chunk_context = context
                        if len(code_chunks) > 1:
                            chunk_context += f" (Part {i+1} of {len(code_chunks)})"
                            
                        cell_data = {
                            "id": str(uuid.uuid4()),
                            "embedding": None,
                            "code": chunk,
                            "filename": os.path.relpath(notebook_path, self.save_dir),
                            "context": chunk_context
                        }
                        extracted_data.append(cell_data)
            
            return extracted_data
        
        except Exception as e:
            self.logger.error(f"Error extracting from notebook {notebook_path}: {str(e)}")
            return []
    
    def _extract_from_code_file(self, file_path: str) -> List[Dict]:
        """
        Extracts code and documentation from general code files.
        
        Args:
            file_path (str): Path to the code file
            
        Returns:
            List of dictionaries with extracted code and context
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # For shorter files, process as a single chunk
            if content.count('\n') <= self.max_chunk_size:
                chunk_data = {
                    "id": str(uuid.uuid4()),
                    "embedding": None,
                    "code": content,
                    "filename": os.path.relpath(file_path, self.save_dir),
                    "context": f"File: {os.path.basename(file_path)}"
                }
                return [chunk_data]
            
            # For longer files, use logical chunking with size limits
            chunks = self._chunk_file_content(content, file_path)
            
            # Convert chunks to the expected output format
            extracted_data = []
            for i, (code, doc_context) in enumerate(chunks):
                if code.strip():  # Only include non-empty code
                    chunk_data = {
                        "id": str(uuid.uuid4()),
                        "embedding": None,
                        "code": code,
                        "filename": os.path.relpath(file_path, self.save_dir),
                        "context": doc_context or f"File: {os.path.basename(file_path)} (Part {i+1} of {len(chunks)})"
                    }
                    extracted_data.append(chunk_data)
            
            return extracted_data
        
        except Exception as e:
            self.logger.error(f"Error extracting from file {file_path}: {str(e)}")
            return []
    
    def _chunk_code(self, code: str, max_lines: int) -> List[str]:
        """
        Splits code into chunks of reasonable size.
        
        Args:
            code (str): Code to split
            max_lines (int): Maximum number of lines per chunk
            
        Returns:
            List of code chunks
        """
        lines = code.split('\n')
        
        # If code is small enough, return as is
        if len(lines) <= max_lines:
            return [code]
            
        # Otherwise break into chunks
        chunks = []
        for i in range(0, len(lines), max_lines):
            chunk_lines = lines[i:min(i+max_lines, len(lines))]
            chunks.append('\n'.join(chunk_lines))
            
        return chunks
    
    def _chunk_file_content(self, content: str, file_path: str) -> List[Tuple[str, str]]:
        """
        Splits file content into logical chunks with size limits.
        
        Args:
            content (str): File content
            file_path (str): Path to the file
            
        Returns:
            List of (code, context) tuples
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        lines = content.split('\n')
        file_basename = os.path.basename(file_path)
        
        # For small files, just return the whole thing
        if len(lines) <= self.max_chunk_size:
            return [(content, f"File: {file_basename}")]
        
        # For large files, use simple chunking with line numbers as context
        chunks = []
        for i in range(0, len(lines), self.max_chunk_size):
            end_line = min(i + self.max_chunk_size, len(lines))
            chunk_content = '\n'.join(lines[i:end_line])
            context = f"File: {file_basename}, Lines {i+1}-{end_line} of {len(lines)}"
            chunks.append((chunk_content, context))
        
        return chunks