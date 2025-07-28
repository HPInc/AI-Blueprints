import base64
import json
import zipfile
import os
import time
from io import BytesIO
from urllib.parse import urlparse
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import requests
import streamlit as st

# ======================================================================
# IMPORTANT: PASTE YOUR GitHubMarkdownProcessor CLASS HERE
# The class that starts with "class GitHubMarkdownProcessor:" from your
# other notebook cell needs to be included here for the GitHub URL logic to work.
# ======================================================================
class GitHubMarkdownProcessor:
    """
    Processor for extracting and parsing Markdown files from GitHub repositories.
    """

    def __init__(
        self,
        repo_url: str,
        access_token: Optional[str] = None,
        save_dir: str = "./parsed_repo",
    ):
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
        parsed_url = urlparse(self.repo_url)
        path_parts = parsed_url.path.strip("/").split("/")
        if len(path_parts) < 2:
            return None, None, "Invalid GitHub URL format."
        return path_parts[0], path_parts[1], None

    def check_repo(self) -> str:
        owner, name, error = self.parse_url()
        if error:
            return error
        url = f"https://api.github.com/repos/{owner}/{name}"
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            repo_data = response.json()
            return "private" if repo_data.get("private") else "public"
        elif response.status_code == 404:
            return "Repository is inaccessible. Please authenticate."
        else:
            return f"Error: {response.status_code}, {response.text}"

    def extract_md_files(self) -> Tuple[Optional[Dict], Optional[str]]:
        owner, name, error = self.parse_url()
        if error:
            return None, error
        url = f"https://api.github.com/repos/{owner}/{name}"
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        response = requests.get(url, headers=headers)
        if not response.ok:
            return None, f"Error: {response.status_code}, {response.text}"
        default_branch = response.json().get("default_branch", "main")
        tree_url = (
            f"https://api.github.com/repos/{owner}/{name}/git/trees/{default_branch}?recursive=1"
        )
        tree_response = requests.get(tree_url, headers=headers)
        if not tree_response.ok:
            return None, f"Error: {tree_response.status_code}, {tree_response.text}"
        dir_structure = {}
        for item in tree_response.json().get("tree", []):
            path = item["path"]
            if item["type"] != "blob" or not path.endswith(".md"):
                continue
            content_url = (
                f"https://api.github.com/repos/{owner}/{name}/contents/{path}"
            )
            content_response = requests.get(content_url, headers=headers)
            if not content_response.ok:
                continue
            file_data = content_response.json()
            try:
                content = base64.b64decode(file_data["content"]).decode("utf-8")
            except Exception as e:
                content = f"Error decoding content: {e}"
            parts = path.split("/")
            current = dir_structure
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = content
        return dir_structure, None

    def run(self) -> Dict[str, str]:
        visibility = self.check_repo()
        if visibility == "Repository is inaccessible. Please authenticate.":
            raise PermissionError("Cannot access repository. Check your access token.")
        structure, error = self.extract_md_files()
        if error:
            raise RuntimeError(f"Markdown extraction failed: {error}")
        raw_data = {}
        def process_structure(
            structure: Dict[str, Union[str, dict]], path: str = ""
        ) -> None:
            for name, content in structure.items():
                current_path = os.path.join(path, name)
                if isinstance(content, dict):
                    process_structure(content, current_path)
                else:
                    raw_data[current_path] = content
        process_structure(structure)
        return raw_data

# --- Helper Function to Set Background & Minimal Styling ---
def set_styles():
    st.markdown(
        """
    <style>
        [data-testid=\"stHeader\"] {
            background-color: rgba(0, 0, 0, 0);
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

# --- Page Configuration & UI ---
st.set_page_config(
    page_title="Markdown Corrector AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)
set_styles()

left_col, mid_col, right_col = st.columns([1, 4, 1])

with mid_col:
    st.title("📚 Markdown Grammar Corrector")
    st.markdown(
        "Enter a public GitHub repository URL or upload files to correct grammar."
    )

    with st.expander("⚙️ API Configuration"):
        api_url = st.text_input(
            "MLflow Model /invocations URL",
            value="https://localhost:55919/invocations",
        )

    st.subheader("Choose Input Method")
    tab1, tab2 = st.tabs(["🔗 GitHub URL", "📁 Upload Files"])

    files_to_process = {}
    input_description = ""

    with tab1:
        repo_url = st.text_input("Public GitHub Repository URL")
        if st.button(
            "🚀 Correct from URL", key="url_button", use_container_width=True, disabled=not repo_url
        ):
            with st.spinner(
                "Fetching repository files... This may take a moment."
            ):
                try:
                    github_token = os.getenv("GITHUB_ACCESS_TOKEN")
                    processor = GitHubMarkdownProcessor(
                        repo_url=repo_url, access_token=github_token
                    )
                    files_to_process = processor.run()
                    input_description = repo_url
                except Exception as e:
                    st.error(f"Failed to fetch repository: {e}")
                    st.stop()

    with tab2:
        uploaded_files = st.file_uploader(
            "Upload .md or .zip files.",
            type=["md", "zip"],
            accept_multiple_files=True,
        )
        if st.button(
            "🚀 Correct Uploaded Files",
            key="file_button",
            use_container_width=True,
            disabled=not uploaded_files,
        ):
            for uploaded_file in uploaded_files:
                if uploaded_file.name.endswith('.zip'):
                    with zipfile.ZipFile(uploaded_file, 'r') as z:
                        for filename in z.namelist():
                            if filename.endswith('.md') and not filename.startswith('__MACOSX/'):
                                with z.open(filename) as f:
                                    files_to_process[filename] = f.read().decode("utf-8")
                elif uploaded_file.name.endswith('.md'):
                    files_to_process[uploaded_file.name] = uploaded_file.getvalue().decode("utf-8")
            input_description = f"{len(uploaded_files)} uploaded file(s)"

if files_to_process:
    if not api_url:
        st.error("Please provide the MLflow API URL.")
        st.stop()

    with st.spinner(
        "Warming up the AI model... This can take a few minutes on the first run."
    ):
        try:
            warmup_df = pd.DataFrame([
                {"repo_url": None, "files": {"warmup.md": "hello"}}
            ])
            warmup_payload = {"dataframe_split": warmup_df.to_dict("split")}
            requests.post(
                api_url, json=warmup_payload, timeout=300, verify=False
            )
        except requests.exceptions.ReadTimeout:
            st.info("Model finished warming up. Starting corrections...")
        except Exception as e:
            st.warning(f"Warm-up call failed, proceeding anyway: {e}")

    st.session_state["corrected_files"] = {}
    st.session_state["original_files"] = files_to_process
    st.session_state["metric_list"] = []

    total_files = len(files_to_process)
    progress_bar = st.progress(0, text="Starting correction process...")
    start_time = time.time()

    for i, (filename, content) in enumerate(files_to_process.items()):
        progress_text = f"Processing file {i+1} of {total_files}: {filename}"
        progress_bar.progress((i + 1) / total_files, text=progress_text)
        try:
            single_file_dict = {filename: content}
            input_df = pd.DataFrame([
                {"repo_url": None, "files": single_file_dict}
            ])
            payload = {"dataframe_split": input_df.to_dict("split")}

            res = requests.post(
                api_url, json=payload, timeout=300, verify=False
            )
            res.raise_for_status()

            response_data = res.json()["predictions"][0]
            if isinstance(response_data, str):
                response_data = json.loads(response_data)

            st.session_state["corrected_files"].update(
                response_data.get("corrected", {})
            )
            st.session_state["metric_list"].append(
                response_data.get("evaluation_metrics", {})
            )

        except requests.exceptions.RequestException as e:
            st.warning(f"Skipped {filename} due to an API error: {e}")
            st.session_state["corrected_files"][filename] = content
            continue

    final_metrics = {}
    if st.session_state["metric_list"]:
        valid_metrics = [m for m in st.session_state["metric_list"] if m]
        if valid_metrics:
            df_metrics = pd.DataFrame(valid_metrics)
            final_metrics = df_metrics.mean().to_dict()

    st.session_state["evaluation_metrics"] = final_metrics
    st.session_state["response_time"] = time.time() - start_time
    st.session_state["last_input_description"] = input_description
    progress_bar.empty()
    st.rerun()

if "corrected_files" in st.session_state:
    with mid_col:
        st.success(
            f"Successfully processed: {st.session_state['last_input_description']}"
        )

        st.subheader("📊 Performance & Evaluation")
        st.metric(
            label="Total Processing Time",
            value=f"{st.session_state['response_time']:.2f} s",
        )

        eval_metrics = st.session_state.get("evaluation_metrics", {})
        if eval_metrics:
            metric_cols = st.columns(len(eval_metrics))
            for i, (name, value) in enumerate(eval_metrics.items()):
                formatted_value = (
                    f"{value:.4f}" if isinstance(value, float) else str(value)
                )
                metric_cols[i].metric(
                    label=name.replace('_', ' ').title(), value=formatted_value
                )
        st.divider()

        st.subheader("📁 Corrected Markdown Files")
