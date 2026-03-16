import streamlit as st
import os
import requests
import tempfile
import base64
import json
from pathlib import Path
from typing import Optional

# Disable SSL warnings for localhost
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Agentic Audio RAG", page_icon="🎧", layout="centered")

# --- Enhanced Custom Styling ---
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem !important;
            max-width: 900px !important;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            background: white;
        }

        .stApp {
            background: white;
        }

        /* Main Title Styling */
        .main-title {
            text-align: center;
            color: black;
            font-weight: 700;
            font-size: 3rem;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            animation: fadeIn 1s ease-in;
        }

        .subtitle {
            text-align: center;
            color: #666666;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            font-weight: 300;
        }

        /* Card Styling */
        .info-card {
            background: linear-gradient(145deg, #ffffff, #f8f9fa);
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            margin: 20px 0;
            border: 1px solid rgba(255,255,255,0.8);
        }

        /* Button Styling */
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            font-size: 18px !important;
            font-weight: 600 !important;
            border-radius: 12px !important;
            padding: 12px 32px !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
        }

        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
        }

        /* File Uploader Styling */
        .uploadedFile {
            border-radius: 10px !important;
            border: 2px solid #667eea !important;
        }

        /* Answer Card */
        .answer-card {
            background: linear-gradient(145deg, #ffffff, #f8f9fa);
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            margin: 15px 0;
            border-left: 6px solid #667eea;
            transition: all 0.3s ease;
        }

        .answer-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.2);
            border-left-color: #764ba2;
        }

        .answer-card h4 {
            color: #2C3E50;
            margin-bottom: 10px;
            font-weight: 600;
        }

        /* Evidence Items */
        .evidence-item {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 15px 20px;
            border-radius: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }

        .evidence-item:hover {
            transform: translateX(5px);
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15);
        }

        /* Timestamp Badge */
        .timestamp-badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
            margin-right: 10px;
        }

        /* Logo Container */
        .logo-container {
            display: flex;
            justify-content: space-around;
            align-items: center;
            margin-bottom: 2rem;
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }

        img[alt="HP Logo"],
        img[alt="AI Studio Logo"],
        img[alt="Z by HP Logo"] {
            width: 80px !important;
            height: auto !important;
            transition: all 0.3s ease;
        }

        img[alt="HP Logo"]:hover,
        img[alt="AI Studio Logo"]:hover,
        img[alt="Z by HP Logo"]:hover {
            transform: scale(1.1);
        }

        /* Section Headers */
        .section-header {
            color: #2C3E50;
            font-weight: 600;
            font-size: 1.3rem;
            margin: 20px 0 15px 0;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        hr {
            border-color: rgba(0,0,0,0.1) !important;
            margin: 2rem 0 !important;
        }
        /* Progress Bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# --- Logo Section ---
def uri_from(path: Path) -> str:
    """Convert image file to base64 data URI"""
    return (
        f"data:image/{path.suffix[1:].lower()};base64,"
        + base64.b64encode(path.read_bytes()).decode()
    )


static = Path("static")
hp_uri = uri_from(static / "HP-Logo.png")
ais_uri = uri_from(static / "AIS-logo.png")
zhp_uri = uri_from(static / "Z-logo.png")

st.markdown(
    f"""
    <div class="logo-container">
        <img src="{hp_uri}" alt="HP Logo">
        <img src="{ais_uri}" alt="AI Studio Logo">
        <img src="{zhp_uri}" alt="Z by HP Logo">
    </div>
""",
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown(
    """
    <h1 class="main-title">🎧 Agentic Audio RAG</h1>
    <p class="subtitle">Upload audio/video files and ask questions powered by AI</p>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# MLflow API Configuration
# ─────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")

st.sidebar.markdown(
    """
**How to use:**
1. Upload an audio or video file
2. Wait for processing (embedding generation)
3. Ask questions about the content
4. View AI-generated answers with timestamps

**Supported Formats:**
- Audio: MP3, WAV, OGG, FLAC, M4A
- Video: MP4, MOV, AVI, MKV, WEBM
"""
)

MLFLOW_ENDPOINT = st.sidebar.text_input(
    "MLflow Model Endpoint URL",
    value="http://localhost:5002/invocations",
    help="Enter the /invocations endpoint of your deployed model",
)

REQUEST_TIMEOUT = st.sidebar.number_input(
    "Request Timeout (seconds)",
    value=3600,
    min_value=60,
    max_value=7200,
    help="Timeout for model requests. Increase for large audio files or slow inference.",
)

# Initialize session state
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "temp_path" not in st.session_state:
    st.session_state.temp_path = None


# ─────────────────────────────────────────────────────────────
# File Upload Section
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<p class="section-header">📁 Upload Audio/Video</p>', unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an audio or video file",
    type=["mp3", "wav", "ogg", "flac", "m4a", "mp4", "mov", "avi", "mkv", "webm"],
    help="Upload a media file to analyze",
)

# Handle file upload
if uploaded_file is not None:
    if uploaded_file.name != st.session_state.uploaded_file_name:
        # New file uploaded
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.file_processed = False
        st.session_state.file_id = uploaded_file.name
        st.session_state.qa_history = []

        st.info(
            f"📎 File loaded: **{uploaded_file.name}** ({uploaded_file.size / 1024 / 1024:.2f} MB)"
        )

        # Process the file
        with st.spinner("🎧 Processing audio and generating embeddings..."):
            try:
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(uploaded_file.name).suffix
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_path = tmp.name

                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Converting audio format...")
                progress_bar.progress(25)

                status_text.text("Generating CLAP embeddings...")
                progress_bar.progress(50)

                # Make API call to process audio
                payload = {
                    "inputs": [
                        {
                            "audio_path": temp_path,
                            "question": "Initialize",  # Dummy question to trigger processing
                            "file_id": st.session_state.file_id,
                        }
                    ]
                }

                status_text.text("Indexing audio segments...")
                progress_bar.progress(75)

                response = requests.post(
                    MLFLOW_ENDPOINT, json=payload, verify=False, timeout=REQUEST_TIMEOUT
                )

                status_text.text("Complete!")
                progress_bar.progress(100)

                if response.status_code == 200:
                    st.session_state.file_processed = True
                    st.session_state.temp_path = temp_path  # Store for later cleanup
                    st.success(
                        "✅ Audio processed successfully! You can now ask questions."
                    )
                else:
                    st.error(f"❌ Processing failed: {response.status_code}")
                    st.session_state.file_processed = False
                    # Clean up temp file only on failure
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.session_state.file_processed = False
    else:
        # Same file - already processed
        if st.session_state.file_processed:
            st.success(f"✅ File ready: **{uploaded_file.name}**")


# ─────────────────────────────────────────────────────────────
# Question-Answer Section
# ─────────────────────────────────────────────────────────────
if st.session_state.file_processed:
    st.markdown("---")
    st.markdown(
        '<p class="section-header">❓ Ask Questions</p>', unsafe_allow_html=True
    )

    with st.form("question_form"):
        question = st.text_area(
            "Enter your question about the audio",
            height=100,
            placeholder="e.g., What was the main topic discussed?",
        )

        submitted = st.form_submit_button("🔍 Get Answer")

    if submitted and question.strip():
        with st.spinner("🤔 Analyzing audio and generating answer..."):
            try:
                payload = {
                    "inputs": [
                        {"question": question, "file_id": st.session_state.file_id}
                    ]
                }

                response = requests.post(
                    MLFLOW_ENDPOINT, json=payload, verify=False, timeout=REQUEST_TIMEOUT
                )

                if response.status_code == 200:
                    result = response.json()["predictions"][0]

                    # Add to history
                    st.session_state.qa_history.insert(
                        0,
                        {
                            "question": question,
                            "answer": result.get("answer", ""),
                            "evidence": result.get("evidence", []),
                            "from_memory": result.get("from_memory", False),
                        },
                    )

                else:
                    st.error(f"❌ Request failed: {response.status_code}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


# ─────────────────────────────────────────────────────────────
# Display Q&A History
# ─────────────────────────────────────────────────────────────
if st.session_state.qa_history:
    st.markdown("---")
    st.markdown(
        '<p class="section-header">💬 Conversation History</p>', unsafe_allow_html=True
    )

    for idx, qa in enumerate(st.session_state.qa_history):
        cache_badge = "🔄 (from cache)" if qa.get("from_memory") else ""

        st.markdown(
            f"""
            <div class="answer-card">
                <h4>Q: {qa['question']} {cache_badge}</h4>
                <p style="color: #2C3E50; font-size: 1.05rem; line-height: 1.6;">
                    <strong>A:</strong> {qa['answer']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Display evidence with timestamps
        if qa.get("evidence"):
            with st.expander(f"🔍 View Evidence ({len(qa['evidence'])} segments)"):
                for i, ev in enumerate(qa["evidence"], 1):
                    start_s = ev.get("start_s", 0)
                    end_s = ev.get("end_s", 0)
                    score = ev.get("score", 0)

                    # Format timestamps
                    start_mm_ss = f"{int(start_s // 60):02d}:{int(start_s % 60):02d}"
                    end_mm_ss = f"{int(end_s // 60):02d}:{int(end_s % 60):02d}"

                    st.markdown(
                        f"""
                        <div class="evidence-item">
                            <span class="timestamp-badge">{start_mm_ss} - {end_mm_ss}</span>
                            <strong>Relevance:</strong> {score:.2%}<br>
                            <strong>File:</strong> {ev.get('file_name', 'Unknown')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # Clear history button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.qa_history = []
            # Clean up temp file when clearing history
            if st.session_state.get("temp_path"):
                try:
                    os.unlink(st.session_state.temp_path)
                except:
                    pass
                st.session_state.temp_path = None
            st.rerun()

elif st.session_state.file_processed:
    st.info("💡 Ask your first question above to get started!")

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.write("---")
st.write("Built with ❤️ using HP AI Studio")
