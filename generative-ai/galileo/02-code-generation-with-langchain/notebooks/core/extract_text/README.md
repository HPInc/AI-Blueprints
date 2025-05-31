# GitHub Repository Extractor

This directory contains the implementation of the GitHub Repository Extractor:

1. `github_repository_extractor.py`

## Approach

The extractor implements several strategies to extract context from the repository:

### 1. File Size Limitations
- Maximum file size parameter (`max_file_size_kb`, default: 500KB)
- Files exceeding this limit are skipped

### 2. Pattern-Based Exclusion
- Automatic certain of problematic file types:
  - Minified files (*.min.js, *.min.css)
  - Bundle files with hash suffixes (index-[hash].js)
  - Node modules and dist directories

### 3. Intelligent Code Chunking
- Breaking large files into smaller chunks (`max_chunk_size`, default: 100 lines)
- Preserving context with line number information
- Maintaining logical code structure where possible

### 4. Context
- Adding file section information to chunks
- Including line number references in context

## Usage

```python
# Original extractor (may cause context window overflow with large files)
extractor = GitHubRepositoryExtractor(
    repo_url="https://github.com/username/repo",
    verbose=False
)

extractor = GitHubRepositoryExtractor(
    repo_url="https://github.com/username/repo",
    verbose=False,
    max_file_size_kb=500,  # Skip files larger than 500KB
    max_chunk_size=100,    # Break large files into chunks of 100 lines
    supported_extensions=('.py', '.ipynb', '.md', '.txt', '.json')  # Focus on these types
)
```

## File Filtering Examples

The extractor automatically filters out files like:

- `dist/assets/index-Cc4mgDJj.js` (bundled file with hash suffix)
- `build/main.min.js` (minified JavaScript)
- `node_modules/react/index.js` (library dependency)
- `vendor.js` (third-party bundle)

This filtering ensures that the extractor only processes files that can be effectively embedded and used for context generation without exceeding token limits.