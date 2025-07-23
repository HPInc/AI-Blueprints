import os
import tempfile
import shutil
import re
import json
import logging
from pathlib import Path
from git import Repo

# Use the same logger as the notebook for consistent output
logger = logging.getLogger("multimodal_rag_logger")

IMAGE_REGEX = re.compile(r'!\[.*?\]\((.+?)\)')

def _clone_wiki_repo(pat: str, org: str, project: str, wiki_id: str) -> Path:
    """
    Clones the Azure DevOps Git-based wiki to a temporary directory.
    Returns the path to the temporary local repo.
    """
    url = f"https://oauth2:{pat}@dev.azure.com/{org}/{project}/_git/{wiki_id}"
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Cloning wiki '{wiki_id}' to temporary directory: {temp_dir}")
    try:
        Repo.clone_from(url, temp_dir)
        return Path(temp_dir)
    except Exception as e:
        logger.error(f"Failed to clone wiki repository: {e}")
        # Clean up the temp dir on failure before raising the error
        shutil.rmtree(temp_dir)
        raise

def _read_markdown_pages(root_dir: Path) -> dict:
    """Reads all .md files from the repo into a dictionary."""
    pages = {}
    logger.info("Scanning for Markdown files...")
    for full_path in root_dir.rglob('*.md'):
        rel_path = full_path.relative_to(root_dir).as_posix()
        text = full_path.read_text(encoding='utf-8')
        raw_imgs = IMAGE_REGEX.findall(text)
        # Normalize image paths to just the filename for easier lookup
        imgs = [Path(i).name for i in raw_imgs]
        pages[rel_path] = {"content": text, "images": imgs}
    logger.info(f"→ Found {len(pages)} Markdown pages.")
    return pages

def _copy_all_images(pages: dict, repo_dir: Path, output_images_dir: Path):
    """Copies every referenced image into a single flat 'images/' folder."""
    output_images_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Copying referenced images to {output_images_dir}...")
    seen = set()
    total_copied = 0
    for info in pages.values():
        for img_name in info["images"]:
            if not img_name or img_name in seen:
                continue
            
            # Find the first occurrence of the image file in the entire cloned repo
            found_path = next(repo_dir.rglob(f"**/{img_name}"), None)
            
            if found_path:
                dst = output_images_dir / img_name
                shutil.copy2(found_path, dst)
                # logger.info(f"  • Copied {img_name}")
                seen.add(img_name)
                total_copied += 1
            else:
                 logger.warning(f"  • Image reference not found in repo: {img_name}")

    logger.info(f"→ {total_copied} unique images copied.")

def run_wiki_clone(pat: str, org: str, project: str, wiki_id: str, output_dir: Path):
    """
    Clones an ADO Wiki, extracts content and images, and saves them to a structured format.

    Args:
        pat (str): Azure DevOps Personal Access Token.
        org (str): ADO Organization name.
        project (str): ADO Project name.
        wiki_id (str): The identifier of the wiki (e.g., 'MyProject.wiki').
        output_dir (Path): The directory to save the final output ('wiki_flat_structure.json' and 'images/').
    """
    if not pat:
        raise ValueError("Azure DevOps PAT cannot be empty. Please set the AZURE_DEVOPS_PAT environment variable.")

    repo_dir = None
    try:
        # 1. Clone the wiki repo to a temporary location
        repo_dir = _clone_wiki_repo(pat, org, project, wiki_id)

        # 2. Read all Markdown pages from the cloned repo
        pages_info = _read_markdown_pages(repo_dir)

        # 3. Define final output paths and ensure directories exist
        output_dir.mkdir(parents=True, exist_ok=True)
        output_images_dir = output_dir / "images"
        output_json_path = output_dir / "wiki_flat_structure.json"

        # 4. Copy all referenced images to the final output location
        _copy_all_images(pages_info, repo_dir, output_images_dir)

        # 5. Assemble the flat list for the JSON structure
        logger.info("Assembling flat JSON structure...")
        flat_list = [
            {"path": path, "content": info["content"], "images": info["images"]}
            for path, info in pages_info.items()
        ]

        # 6. Write the data to wiki_flat_structure.json
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(flat_list, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Wiki data successfully cloned to {output_dir}")

    finally:
        # 7. Clean up the temporary cloned repo directory
        if repo_dir and repo_dir.exists():
            shutil.rmtree(repo_dir)
            logger.info(f"Cleaned up temporary directory: {repo_dir}")