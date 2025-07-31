import streamlit as st
import shutil
import requests
import base64
import os
import json
from pathlib import Path
from typing import List
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
    page_title="Agentic Feedback Analyzer",
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
st.markdown('<div class="gradient-header"><h2>🤖 Agentic Feedback Analyzer 🤖</h2></div>', unsafe_allow_html=True)

# ------------------------- SIDEBAR CONFIG -------------------------
st.sidebar.title("⚙️ Configuration")

st.sidebar.markdown("""
**Instructions:**
1. Enter your model's `/invocations` endpoint URL.
2. Fill out the topic, question, and input text.
3. Click **Run Analysis** to receive an AI-generated answer.

**Example URL:** `https://localhost:5000/invocations`
""")

endpoint_url = st.sidebar.text_input("MLflow Model Endpoint URL", key="endpoint")
endpoint_url = 'https://localhost:5000/invocations'

if endpoint_url and not endpoint_url.strip().lower().startswith("https://"):
    st.sidebar.error("Endpoint must start with https://")

# ------------------------- INPUTS -------------------------
with st.form("inference_form"):
    topic = st.text_area("📂 Topic", height=80, key="topic")
    question = st.text_area("❓ Question", height=100, key="question")
    uploaded_files = st.file_uploader(
        "Upload Document Files", accept_multiple_files=True,
        type=["txt", "csv", "pdf", "docx", "xlsx", "md"]
    )

    submitted = st.form_submit_button("🔢 Run Analysis")

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
    

# File Type Mapping
supported_extensions = {
    ".txt": SafeTextLoader,
    ".csv": lambda path: CSVLoader(path, encoding="utf-8", csv_args={"delimiter": ","}),
    ".xlsx": UnstructuredExcelLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
}


def process_files(files: List) -> str:
    all_docs = []
    temp_dir = Path(".tmp")
    temp_dir.mkdir(exist_ok=True)

    try:
        for file in files:
            suffix = Path(file.name).suffix.lower()
            loader_class = supported_extensions.get(suffix)

            if not loader_class:
                st.warning(f"⚠️ Unsupported file type: {file.name}")
                continue

            try:
                # Save uploaded file to disk
                temp_path = temp_dir / file.name
                with open(temp_path, "wb") as f:
                    f.write(file.getvalue())

                # Normalize path for loaders (especially for Windows)
                resolved_path = temp_path.resolve()

                # Instantiate and load document
                loader = loader_class(str(resolved_path))
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                st.error(f"❌ Failed to load {file.name}: {e}")
    finally:
        # Always attempt to clean up the temporary directory
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_err:
                st.warning(f"⚠️ Failed to clean up temp directory: {cleanup_err}")

    return '\n\n'.join(doc.page_content for doc in all_docs)

# ------------------------- API CALL -------------------------
if submitted:
    if not endpoint_url or not topic or not question or not uploaded_files:
        st.error("Please fill in all required fields.")
        input_text = process_files(uploaded_files)
    else:
        with st.spinner("Analyzing with AI..."):
            input_text = process_files(uploaded_files)

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
                    # endpoint_url.strip(),
                    "https://27eb5d57a425.ngrok.app/invocations",
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
