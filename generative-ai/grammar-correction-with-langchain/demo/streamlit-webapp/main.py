import base64
import difflib
import json
import zipfile
import os
import time
from io import BytesIO
from urllib.parse import urlparse
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import re

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# ==============================================================================
# INITIALIZE SESSION STATE
# ==============================================================================
# This ensures that all keys are always available in the session state, preventing KeyErrors. 

if "corrected_files" not in st.session_state:
    st.session_state.corrected_files = {}

if "original_files" not in st.session_state:
    st.session_state.original_files = {}

if "metric_list" not in st.session_state:
    st.session_state.metric_list = []

if "evaluation_metrics" not in st.session_state:
    st.session_state.evaluation_metrics = {}

if "response_time" not in st.session_state:
    st.session_state.response_time = 0

if "last_input_description" not in st.session_state:
    st.session_state.last_input_description = "" 

# ==============================================================================
# LARGE FILE SPLIT HELPER FUNCTION
# ==============================================================================

LARGE_FILE_CHARACTER_LIMIT = 40000

def split_text_if_too_large(text: str, max_size: int = LARGE_FILE_CHARACTER_LIMIT) -> list[str]:
    """
    If text exceeds max_size, recursively splits it in half at the best possible
    break point (paragraph, sentence, etc.) until all parts are under the limit.
    """
    # Base Case: The text is small enough, no need to split.
    if len(text) <= max_size:
        return [text]

    # Aim for the middle of the text as the split point.
    ideal_split_point = len(text) // 2
    
    # Define the preferred delimiters in order from best to worst.
    split_delimiters = ['\n\n', '. ', '? ', '! ', '\n', ' ']
    
    split_pos = -1 # This will hold the position where we'll split the text.
    
    # Search backwards from the ideal split point to find the best delimiter.
    for delimiter in split_delimiters:
        # Look for the last occurrence of the delimiter before the ideal split point.
        pos = text.rfind(delimiter, 0, ideal_split_point)
        
        if pos != -1:
            # Found a good delimiter
            split_pos = pos + len(delimiter)
            break # Stop searching, since we found the best possible option.
            
    if split_pos == -1:
        split_pos = ideal_split_point

    part1 = text[:split_pos].strip()
    part2 = text[split_pos:].strip()
    
    # Recurse on both halves and combine the results.
    return split_text_if_too_large(part1, max_size) + split_text_if_too_large(part2, max_size)

# ==============================================================================
# GITHUB EXTRACTOR CLASS
# ==============================================================================
class GitHubMarkdownProcessor:
    """
    Processor for extracting and parsing Markdown files from GitHub repositories.
    """
    def __init__(
        self,
        repo_url: str,
        access_token: Optional[str] = None,
        save_dir: str = "./parsed_repo",
        timeout: int = 20,
    ):
        self.repo_url = repo_url
        self.access_token = access_token
        self.save_dir = save_dir
        self.timeout = timeout
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
        response = requests.get(url, headers=headers, timeout=self.timeout)
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
        
        response = requests.get(url, headers=headers, timeout=self.timeout)
        if not response.ok:
            return None, f"Error: {response.status_code}, {response.text}"
        
        default_branch = response.json().get("default_branch", "main")
        tree_url = (
            f"https://api.github.com/repos/{owner}/{name}/git/trees/{default_branch}?recursive=1"
        )
        
        tree_response = requests.get(tree_url, headers=headers, timeout=self.timeout)
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
            
            try:
                content_response = requests.get(content_url, headers=headers, timeout=self.timeout)
                content_response.raise_for_status()
                
                file_data = content_response.json()
                content = base64.b64decode(file_data["content"]).decode("utf-8")
            except requests.exceptions.Timeout:
                print(f"Warning: Timed out while fetching '{path}'. Skipping this file.")
                continue
            except Exception as e:
                print(f"Warning: Could not process '{path}' due to an error: {e}. Skipping file.")
                continue
            
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

# ==============================================================================
# STYLING AND UI
# ==============================================================================
def set_bg_hack(file_path: str):
    """
    A function to set a background image from a local file.
    Takes a string path as input.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: top center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
            background-color: rgba(0, 0, 0, 0);
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)

        other_styles = """
        <style>
        /* --- CSS for Modern HTML Diff --- */
        table.diff {
            font-family: Menlo, Monaco, 'Courier New', monospace;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9em;
            width: 100%;
        }
        .diff_header { background-color: #262730; color: #FAFAFA; font-weight: 600; padding: 8px 12px; }
        .diff_add { background-color: #204020; }
        .diff_chg { background-color: #4d4d20; }
        .diff_sub { background-color: #4d2020; }
        td { padding: 5px 8px; vertical-align: top; white-space: pre-wrap; word-wrap: break-word; }
        td[id^="from"], td[id^="to"] { color: #888 !important; font-weight: 500; }
        
        /* --- CSS for Wide, Centered Diff Component --- */
        div[data-testid="stHtml"] iframe {
            width: 90vw;
            max-width: 1400px;
            position: relative;
            left: 50%;
            transform: translateX(-50%);
            border: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        </style>
        """
        st.markdown(other_styles, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error(f"Asset not found. Please check the path: {file_path}")

# --- PAGE CONFIGURATION & MAIN APP LAYOUT ---
st.set_page_config(page_title="Markdown Corrector AI", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

try:
    image_path = Path(__file__).resolve().parent.parent / "assets" / "background.png"
    set_bg_hack(str(image_path))
except Exception:
    # Fallback for environments where path logic might differ
    pass

_, mid_col, _ = st.columns([1, 4, 1])

with mid_col:
    st.title("📚 Markdown Grammar Corrector")
    st.markdown("Enter a public GitHub repository URL or upload files to correct grammar.")
    
    with st.expander("⚙️ API Configuration"):
        api_url = st.text_input("MLflow Model /invocations URL", value="https://localhost:55919/invocations")

    st.subheader("Choose Input Method")
    tab1, tab2 = st.tabs(["🔗 GitHub URL", "📁 Upload Files"])
    
    files_to_process = {}
    input_description = ""

    with tab1:
        repo_url = st.text_input("GitHub Repository URL")
        # --- MODIFIED: Added input field for the token ---
        github_token = st.text_input(
            "GitHub Access Token",
            type="password",
            help="Enter your Personal Access Token (PAT) for private repos or to avoid rate limits."
        )
        if st.button("🚀 Correct from URL", key="url_button", use_container_width=True, disabled=not repo_url):
            with st.spinner("Fetching repository files..."):
                try:
                    # --- MODIFIED: Use the token from the input field ---
                    processor = GitHubMarkdownProcessor(repo_url=repo_url, access_token=github_token)
                    files_to_process = processor.run()
                    input_description = repo_url
                except Exception as e:
                    st.error(f"Failed to fetch repository: {e}")
                    st.stop()
                    
    with tab2:
        uploaded_files = st.file_uploader("Upload .md or .zip files", type=["md", "zip"], accept_multiple_files=True)
        if st.button("🚀 Correct Uploaded Files", key="file_button", use_container_width=True, disabled=not uploaded_files):
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

# ==============================================================================
# PROCESSING LOGIC
# ==============================================================================
if files_to_process:
    with mid_col: 
        if not api_url:
            st.error("Please provide the MLflow API URL.")
            st.stop()
        
        # --- MODIFIED: Improved warmup with clear error handling ---
        with st.spinner("Initializing the model... This can take a few minutes on the first run."):
            try:
                # Use a string for the warmup payload to match the real calls
                warmup_df = pd.DataFrame([{"repo_url": None, "files": "This is a warmup."}])
                warmup_payload = {"dataframe_split": warmup_df.to_dict("split")}
                # Check the response and raise an error if it fails
                requests.post(api_url, json=warmup_payload, timeout=180, verify=False).raise_for_status()
            except Exception as e:
                st.error(f"Failed to initialize the model. Please check the API URL and ensure the backend is running correctly. Error: {e}")
                st.stop() # Stop the app if the model can't be reached

        st.session_state["corrected_files"] = {}
        st.session_state["original_files"] = files_to_process
        st.session_state["metric_list"] = []
        
        total_files = len(files_to_process)
        progress_bar = st.progress(0, text="Starting correction process...")
        start_time = time.time()

        for i, (filename, content) in enumerate(files_to_process.items()):
            progress_bar.progress((i + 1) / total_files, text=f"Processing file {i+1} of {total_files}: {filename}")
            
            # 1. Split the file content only if it's an anomaly.
            # Most files will result in a list with a single item.
            pieces = split_text_if_too_large(content, max_size=LARGE_FILE_CHARACTER_LIMIT)
            
            corrected_pieces = []
            
            # Only show the spinner if the file was actually split
            spinner_text = f"Processing {filename}..."
            if len(pieces) > 1:
                spinner_text = f"Processing {filename} in {len(pieces)} parts..."

            with st.spinner(spinner_text):
                for piece in pieces:
                    try:
                        # Process each piece (which could be the whole file)
                        input_df = pd.DataFrame([{"repo_url": None, "files": piece}])
                        payload = {"dataframe_split": input_df.to_dict("split")}
                        res = requests.post(api_url, json=payload, timeout=300, verify=False)
                        res.raise_for_status()
                        
                        response_data = res.json()["predictions"][0]
                        if isinstance(response_data, str):
                            response_data = json.loads(response_data)
                        
                        corrected_content_dict = response_data.get("corrected", {})
                        if "corrected_file.md" in corrected_content_dict:
                            corrected_pieces.append(corrected_content_dict["corrected_file.md"])
                        else:
                            corrected_pieces.append(piece)

                        st.session_state["metric_list"].append(response_data.get("evaluation_metrics", {}))

                    except Exception as e:
                        st.warning(f"A part of {filename} failed to process: {e}. Keeping original content for this part.")
                        corrected_pieces.append(piece)
                        continue
            
            # 3. Stitch the corrected pieces back together
            # If the file wasn't split, this just joins a single-item list.
            final_corrected_text = "\n\n".join(corrected_pieces)
            st.session_state["corrected_files"][filename] = final_corrected_text

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

# ==============================================================================
# DISPLAY RESULTS 
# ==============================================================================

METRIC_MAX_SCORES = {
    "grammar_quality_score": 10.0,
    "semantic_similarity": 10.0,
}

if "corrected_files" in st.session_state:
    with mid_col:
        st.success(f"Successfully processed: {st.session_state['last_input_description']}")
        st.subheader("📊 Performance & Evaluation")
        
        all_metrics = {
            "total_processing_time": st.session_state['response_time']
        }
        other_metrics = st.session_state.get("evaluation_metrics", {})
        all_metrics.update(other_metrics)
        
        if all_metrics:
            metric_cols = st.columns(min(len(all_metrics), 5))
            for i, (name, value) in enumerate(all_metrics.items()):
                if isinstance(value, (int, float)):
                    max_score = METRIC_MAX_SCORES.get(name)
                    
                    # --- START OF MODIFICATION ---
                    if max_score:
                        # Logic for grammar_quality_score and semantic_similarity
                        if max_score > 0:
                            percentage = (value / max_score) * 100
                            formatted_value = f"{percentage:.1f}%"
                        else:
                            formatted_value = "N/A"
                    else:
                        # Logic for readability_improvement and time
                        if name == "readability_improvement":
                            # Format the delta as a signed percentage, e.g., "+10.5%" or "-5.2%"
                            formatted_value = f"{value:+.1f}%"
                        else: # Keep original logic for other metrics like 'time'
                            formatted_value = f"{value:.2f}"
                            if "time" in name.lower():
                                formatted_value += " s"
                    # --- END OF MODIFICATION ---
                    
                    formatted_label = name.replace('_', ' ').title()
                    metric_cols[i % 5].metric(label=formatted_label, value=formatted_value)
            
        st.divider()

        st.subheader("📁 Corrected Markdown Files")
        corrected_files = st.session_state.get("corrected_files", {})
        original_files = st.session_state.get("original_files", {})
        
        if corrected_files:
            file_names = sorted(corrected_files.keys())
            selected_file = st.selectbox("📄 Choose a file to compare", file_names)

            if selected_file:
                st.subheader("✨ Side-by-Side Comparison")
                original_text_lines = original_files.get(selected_file, "").splitlines()
                corrected_text_content = corrected_files.get(selected_file, "") # Get current version

                s = difflib.SequenceMatcher(None, original_text_lines, corrected_text_content.splitlines())
                grouped_opcodes = s.get_grouped_opcodes(n=3) # n=3 lines of context

                display_original = []
                display_corrected = []
                has_changes = False
                
                for i, group in enumerate(grouped_opcodes):
                    if i > 0:
                        display_original.append("...")
                        display_corrected.append("...")
                    
                    for tag, i1, i2, j1, j2 in group:
                        if tag != 'equal':
                            has_changes = True
                        display_original.extend(original_text_lines[i1:i2])
                        display_corrected.extend(corrected_text_content.splitlines()[j1:j2])

                if has_changes:
                    differ = difflib.HtmlDiff(wrapcolumn=66)
                    diff_html = differ.make_file(display_original, display_corrected, fromdesc="Original", todesc="Corrected")
                    
                    font_fix_css = "<style> table.diff td { font-family: system-ui, sans-serif !important; font-size: 1.05em !important; } </style>"
                    diff_html = diff_html.replace('</head>', f'{font_fix_css}</head>')
                    diff_html = re.sub(r'<a[^>]*>|</a>', '', diff_html)
                    
                    components.html(diff_html, height=600, scrolling=True)
                else:
                    st.info("✅ No differences found in this file.")

                with st.expander("✍️ Manually Edit Corrected File"):
                    edited_text = st.text_area(
                        label="Make any final changes to the corrected text below. Your edits will be saved for the download.",
                        value=corrected_text_content,
                        height=500,
                        key=f"editor_{selected_file}" # A unique key
                    )

                    # Immediately update the session state with any edits.
                    st.session_state["corrected_files"][selected_file] = edited_text

            st.divider()
            
            # download button logic
            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as z:
                for path, txt in corrected_files.items():
                    z.writestr(path, txt)
            zip_buf.seek(0)
            st.download_button(
                label="📦 Download All Corrected Files",
                data=zip_buf,
                file_name="corrected_markdown.zip",
                mime="application/zip",
                use_container_width=True,
            )