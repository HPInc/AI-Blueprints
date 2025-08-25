import streamlit as st
import shutil
import requests
import base64
import os
import json
from pathlib import Path
from typing import List
import zipfile
import io
from langchain.document_loaders import (
    TextLoader,
    CSVLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
)
from langchain.docstore.document import Document

# Set page config
st.set_page_config(
    page_title="Agentic Github Repo Analyzer",
    page_icon="🔬",
    layout="wide"
)

# ------------------------- CSS STYLING -------------------------

st.markdown(
    "<style>" + open("assets/styles.css").read() + "</style>", 
    unsafe_allow_html=True
)

# ------------------------- LOGOS -------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.image("static/HP-logo.png", width=100)
with col2:
    st.image("static/Z-logo.png", width=100)
with col3:
    st.image("static/AIS-logo.png", width=100)


# ------------------------- HEADER -------------------------
st.markdown('<div class="gradient-header"><h2>🤖 Agentic Github Repo Analyzer 🤖</h2></div>', unsafe_allow_html=True)

# ------------------------- SIDEBAR CONFIG -------------------------
st.sidebar.title("⚙️ Configuration")

st.sidebar.markdown("""
**Instructions:**
1. Enter your model's `/invocations` endpoint URL.
2. Fill out the topic, question, repo url, and folder path.
3. Click **Run Analysis** to receive an AI-generated answer.

**Example URL:** `https://localhost:5000/invocations`
""")

endpoint_url = st.sidebar.text_input("MLflow Model Endpoint URL", key="endpoint")

if endpoint_url and not endpoint_url.strip().lower().startswith("https://"):
    st.sidebar.error("Endpoint must start with https://")

# ------------------------- INPUTS -------------------------
with st.form("inference_form"):
    topic = st.text_area("📂 Topic", height=80, key="topic")
    question = st.text_area("❓ Question", height=100, key="question")
    repo_url = st.text_input("Enter GitHub Repo URL", "https://github.com/HPInc/AI-Blueprints")
    folder_path = st.text_input("Enter Folder Name in Repo", "data-science/classification-with-svm")


    submitted = st.form_submit_button("🔢 Run Analysis")

# ------------------------- REPO DOWNLOADING -------------------------

def download_github_repo(repo_url: str, output_dir: str = "../data/input/downloaded_repo") -> Path:
    """
    Download and extract a public GitHub repository as a zip file.

    Args:
        repo_url (str): The GitHub URL (e.g., https://github.com/HPInc/AI-Blueprints)
        output_dir (str): Directory to extract the repo contents into.

    Returns:
        Path: Path to the extracted root folder.
    """
    # Normalize URL
    repo_url = repo_url.rstrip("/")
    if not repo_url.startswith("https://github.com/"):
        raise ValueError("URL must be a valid GitHub repository URL.")

    # Extract user/repo name
    parts = repo_url.replace("https://github.com/", "").split("/")
    if len(parts) != 2:
        raise ValueError("URL must be in format: https://github.com/owner/repo")

    owner, repo = parts
    zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip"

    print(f"📦 Downloading repo zip from: {zip_url}")
    response = requests.get(zip_url)
    response.raise_for_status()

    # Extract zip contents
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(output_dir)

    # The zip structure is usually: output_dir/repo-main/
    extracted_path = Path(output_dir) / f"{repo}-main"
    print(f"✅ Repository extracted to: {extracted_path.resolve()}")

    return extracted_path


# ------------------------- FILE PROCESSING -------------------------
class SafeTextLoader(TextLoader):
    def load(self) -> list[Document]:
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                with open(self.file_path, encoding=enc) as f:
                    text = f.read()
                return [Document(page_content=text)]
            except Exception:
                continue
        raise ValueError(f"Failed to decode file: {self.file_path}")

class NotebookLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        try:
            with open(self.file_path, encoding="utf-8") as f:
                notebook = json.load(f)
            cells = notebook.get("cells", [])
            source = "\n".join("".join(cell.get("source", [])) for cell in cells)
            return [Document(page_content=source)]
        except Exception as e:
            raise ValueError(f"Failed to load notebook: {self.file_path}: {e}")
    

# File Type Mapping
supported_extensions = {
    ".txt": SafeTextLoader,
    ".csv": lambda path: CSVLoader(path, encoding="utf-8", csv_args={"delimiter": ","}),
    ".xlsx": UnstructuredExcelLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": SafeTextLoader,
    ".json": SafeTextLoader,
    ".yml": SafeTextLoader,
    ".yaml": SafeTextLoader,
    ".ipynb": NotebookLoader,
}


def process_files(input_path: str) -> str:
    all_docs = []

    
    for file_path in Path(input_path).rglob("*"):
        if any(part.startswith(".") and part not in {".", ".."} for part in file_path.parts):
            continue

        if file_path.is_dir():
            continue

        ext = file_path.suffix.lower()
        loader_class = supported_extensions.get(ext)

        if loader_class:
            try:
                loader = loader_class(str(file_path))
                docs = loader.load()
                all_docs.extend(docs)
                print("✅ Loaded %d docs from %s", len(docs), file_path.name)
            except Exception as e:
                print("❌ Failed to load %s: %s", file_path.name, e)
        else:
            print("⚠️ Unsupported file type: %s", file_path.name)

    return '\n\n'.join(doc.page_content for doc in all_docs)

# ------------------------- API CALL -------------------------
if submitted:
    if not endpoint_url or not topic or not question:
        st.error("Please fill in all required fields.")
    else:
        with st.spinner("Getting the Files from the Repo..."):
            repo_name = Path(repo_url.split('/')[-1] + '-main')
            input_path: Path = Path("../data/input/downloaded_repo") / repo_name /  folder_path

            download_github_repo(repo_url)
            input_text = process_files(input_path)
        with st.spinner("Analyzing with AI..."):
            payload = {
                "inputs": [
                    {
                        "topic": topic,
                        "question": question,
                        "input_text": input_text,
                    }
                ],
                "params": {}
            }
            
            try:
                response = requests.post(
                    endpoint_url.strip(),
                    json=payload,
                    verify=False,
                    timeout=600
                )
                response.raise_for_status()
                output = response.json()["predictions"][0]  # Assuming single-record output

                st.markdown("### 📈 Final Answer")
                st.markdown(f"<div class='result-box'>{output['answer']}</div>", unsafe_allow_html=True)
                st.divider()

                with st.expander("🔍 Full Message Trace"):
                    st.json(json.loads(output['messages']))

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
            except Exception as ex:
                st.error(f"Unexpected error: {ex}")
