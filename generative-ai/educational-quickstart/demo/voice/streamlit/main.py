"""
AI Learning Playground — Voice Assistant Demo

Focused Streamlit frontend for the AIStudio-EQ-Voice model.
This app sends requests to the registered VoiceModel via the MLflow invocations endpoint.

Input sent to model:
    question      — Text command (used when no audio is provided)
    audio_base64  — Base64-encoded audio bytes (WAV, MP3, OGG, FLAC)

Pipeline:
    Audio file → base64 encode → VoiceModel → Whisper transcription → LLM → Response

To start the MLflow model server:
    mlflow models serve -m models:/AIStudio-EQ-Voice/1 -p 5002 --no-conda

Then launch this app:
    python -m poetry run streamlit run main.py
"""

import base64
import json
from pathlib import Path

import requests
import streamlit as st

# ───────────────────────────── Page Configuration ─────────────────────────────
st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🎙️",
    layout="wide",
)

# ───────────────────────────── CSS Styling ─────────────────────────────────────
css_path = Path("assets/styles.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ───────────────────────────── Logo Bar ────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    if Path("static/HP-logo.png").exists():
        st.image("static/HP-logo.png", width=100)
with col2:
    if Path("static/Z-logo.png").exists():
        st.image("static/Z-logo.png", width=100)
with col3:
    if Path("static/AIS-logo.png").exists():
        st.image("static/AIS-logo.png", width=100)

# ───────────────────────────── Header ──────────────────────────────────────────
st.markdown(
    '<div class="gradient-header">'
    "<h2>🎙️ Voice Assistant</h2>"
    "<p>Record or upload audio · Whisper large-v3 STT · Llama 3.1 8B LLM · XTTS v2 TTS</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ───────────────────────────── Sidebar ─────────────────────────────────────────
st.sidebar.title("⚙️ Usage")
st.sidebar.markdown(
    """
**Instructions:**
1. Record with your microphone or upload an audio file.
2. Whisper transcribes the audio to text.
3. The LLM generates a response.
4. XTTS v2 speaks the response aloud.
"""
)

endpoint_url = "http://localhost:5002/invocations"


# ───────────────────────────── Helper: API Call ────────────────────────────────


def call_model(audio_base64: str, timeout: int = 600) -> dict:
    """
    Send a POST request to the VoiceModel's invocations endpoint.

    Payload schema — only the columns VoiceModel expects:
        question      (str) — always empty; audio is the only input path
        audio_base64  (str) — base64-encoded audio bytes
    """
    payload = {
        "inputs": [{"question": "", "audio_base64": audio_base64}],
        "params": {},
    }
    try:
        response = requests.post(
            endpoint_url.strip(),
            json=payload,
            verify=False,
            timeout=timeout,
        )
        response.raise_for_status()
        return {"success": True, "data": response.json()["predictions"][0]}
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": (
                "Cannot connect to the model server.\n\n"
                "Start it with:\n"
                "```bash\n"
                "mlflow models serve -m models:/AIStudio-EQ-Voice/1 -p 5002 --no-conda\n"
                "```"
            ),
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out — audio transcription can take 30–90 seconds.",
        }
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}


def render_result(result: dict) -> None:
    """Display the voice assistant's response, including TTS audio if available."""
    if result["success"]:
        output = result["data"]
        st.markdown("### 🤖 Assistant Response")
        st.markdown(
            f"<div class='result-box'>{output['answer']}</div>",
            unsafe_allow_html=True,
        )

        # ── TTS playback ────────────────────────────────────────────────────
        response_audio_b64 = output.get("response_audio", "")
        if response_audio_b64:
            st.markdown("### 🔊 Spoken Response (XTTS v2)")
            audio_bytes = base64.b64decode(response_audio_b64)
            st.audio(audio_bytes, format="audio/wav", autoplay=True)
        # ────────────────────────────────────────────────────────────────────

        st.divider()
        with st.expander("🔍 View Pipeline Details"):
            try:
                st.json(json.loads(output.get("messages", "[]")))
            except json.JSONDecodeError:
                st.text(output.get("messages", ""))
    else:
        st.error(result["error"])


# ───────────────────────────── Voice Input ─────────────────────────────────────
st.markdown("### 🎤 Voice Input")

tab_mic, tab_upload = st.tabs(["🎙️ Record", "📁 Upload"])

with tab_mic:
    st.markdown("Click **Record** to capture your question with the microphone.")
    mic_audio = st.audio_input("Record your question")
    if mic_audio is not None:
        audio_bytes = mic_audio.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        st.info(f"Recorded {len(audio_bytes) / 1024:.1f} KB")
        with st.spinner("Transcribing and generating response..."):
            result = call_model(audio_base64=audio_b64)
        render_result(result)

with tab_upload:
    st.markdown("Upload a WAV, MP3, OGG, or FLAC file to process.")
    audio_file = st.file_uploader(
        "Upload Audio File",
        type=["wav", "mp3", "ogg", "flac"],
        help="Whisper supports WAV, MP3, OGG, FLAC. Larger files take longer to process.",
    )
    if audio_file is not None:
        audio_bytes = audio_file.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        st.info(f"Processing: {audio_file.name} ({len(audio_bytes) / 1024:.1f} KB)")
        if st.button("🎙️ Process", use_container_width=True):
            with st.spinner("Transcribing audio and generating response..."):
                result = call_model(audio_base64=audio_b64)
            render_result(result)
