import os
import tempfile
import shutil
import re
import json
from pathlib import Path
from git import Repo  # pip install GitPython

IMAGE_REGEX = re.compile(r'!\[.*?\]\((.+?)\)')


def clone_wiki_repo(pat, org, project, wiki_id, output_dir=None):
    """
    Clone the Azure DevOps Git-based wiki and return path to local repo.
    """
    url = f"https://oauth2:{pat}@dev.azure.com/{org}/{project}/_git/{wiki_id}"
    # Use a temporary dir or custom one
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    print(f"Cloning wiki to {output_dir}")
    Repo.clone_from(url, output_dir)
    return output_dir

def read_markdown_pages(root_dir):
    """Read all .md files into a dict: rel_path → {content, images}."""
    pages = {}
    print("[2/5] Scanning for Markdown files…")
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if not fn.lower().endswith('.md'):
                continue
            full = Path(dirpath) / fn
            rel  = full.relative_to(root_dir).as_posix()
            print(f"    • Reading {rel}")
            text = full.read_text(encoding='utf-8')
            raw_imgs = IMAGE_REGEX.findall(text)
            # Normalize image paths
            imgs = [Path(i).name for i in raw_imgs]
            pages[rel] = {
                "content": text,
                "images": imgs,
                "children": []  # placeholder for recursive build
            }
    print(f"    → Found {len(pages)} Markdown pages.")
    return pages

def build_recursive_tree(pages):
    """Link up pages into a nested tree based on directory structure."""
    print("[3/5] Building recursive page tree…")
    # First, sort keys so parents come before children
    for path in sorted(pages):
        parent_dir = Path(path).parent.as_posix()
        if parent_dir == '.':
            continue
        # find the nearest ancestor page: e.g. 'Section.md' is parent of 'Section/Subpage.md'
        parts = parent_dir.split('/')
        while parts:
            maybe_parent = '/'.join(parts) + '.md'
            if maybe_parent in pages:
                pages[maybe_parent]["children"].append(path)
                break
            parts.pop()
    # Now extract top-level roots
    roots = [p for p in pages if Path(p).parent == Path('.')]
    print(f"    → {len(roots)} top-level pages found.")
    return roots

def copy_all_images(pages, root_dir, output_images_dir):
    """Copy every referenced image into one flat images/ folder."""
    os.makedirs(output_images_dir, exist_ok=True)
    print(f"[4/5] Copying referenced images to {output_images_dir}…")
    seen = set()
    for info in pages.values():
        for img in info["images"]:
            if img in seen:
                continue
            # find first occurrence under the repo (search entire tree)
            for dirpath, _, fns in os.walk(root_dir):
                if img in fns:
                    src = Path(dirpath) / img
                    dst = Path(output_images_dir) / img
                    print(f"    • {img}")
                    shutil.copy2(src, dst)
                    seen.add(img)
                    break
    print(f"    → {len(seen)} images copied.")

def build_output_structure(pages, roots):
    """Recursively assemble JSON-friendly dicts starting from each root page."""
    def recurse(path):
        node = {
            "content": pages[path]["content"],
            "images": pages[path]["images"],
            "subpages": [recurse(child) for child in pages[path]["children"]]
        }
        return { path: node }

    output = {}
    for root in roots:
        output.update(recurse(root))
    return output


if __name__ == "__main__":
    PAT     = os.getenv("AZURE_DEVOPS_PAT")
    ORG     = os.getenv("AZURE_DEVOPS_ORG", "hpswapps")
    PROJECT = os.getenv("AZURE_DEVOPS_PROJECT", "Phoenix-DS Platform")
    WIKI_ID = os.getenv("AZURE_DEVOPS_WIKI_IDENTIFIER", "Phoenix-DS-Platform.wiki")

    # 1. Clone the wiki
    repo_dir = clone_wiki_repo(PAT, ORG, PROJECT, WIKI_ID)

    # 2. Read all Markdown pages
    pages_info = read_markdown_pages(repo_dir)

    # 3. Copy images
    copy_all_images(pages_info, repo_dir, output_images_dir="data/images")

    # 4. Flatten the pages into a list format
    print("[5/5] Assembling flat JSON structure…")
    flat_list = []
    for path, info in pages_info.items():
        flat_list.append({
            "path": path,
            "content": info["content"],
            "images": info["images"]
        })

    # 5. Write to file
    os.makedirs("data", exist_ok=True)
    with open("data/wiki_flat_structure.json", "w", encoding="utf-8") as f:
        json.dump(flat_list, f, indent=2, ensure_ascii=False)

    # 6. Cleanup
    shutil.rmtree(repo_dir)
    print("Done!  Generated wiki_flat_structure.json and images/ folder.")