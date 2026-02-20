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
    "<p>Speech recognition powered by Whisper large-v3 · LLM responses via LlamaCpp</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ───────────────────────────── Sidebar ─────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown(
    """
**MLflow Server**

Start the voice model server before using this app:

```bash
mlflow models serve \\
  -m models:/AIStudio-EQ-Voice/1 \\
  -p 5002 --no-conda
```

**Pipeline**

1. Upload WAV/MP3/OGG/FLAC audio
2. Whisper transcribes the audio → text
3. LlamaCpp generates a response
4. Both transcript and response are shown

**No microphone?**

Use the text input below to send a command
directly to the LLM, skipping Whisper.
"""
)

endpoint_url = st.sidebar.text_input(
    "API Endpoint URL",
    value="http://localhost:5002/invocations",
    help="The MLflow model server invocations endpoint.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "📚 [Blueprint README](../README.md) | [HP AI Studio](https://hp.com/ai-studio)"
)


# ───────────────────────────── Helper: API Call ────────────────────────────────

def call_model(question: str = "", audio_base64: str = "", timeout: int = 600) -> dict:
    """
    Send a POST request to the VoiceModel's invocations endpoint.

    Payload schema — only the columns VoiceModel expects:
        question      (str) — text fallback
        audio_base64  (str) — base64-encoded audio bytes
    """
    payload = {
        "inputs": [{"question": question, "audio_base64": audio_base64}],
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
    """Display the voice assistant's response."""
    if result["success"]:
        output = result["data"]
        st.markdown("### 🤖 Assistant Response")
        st.markdown(
            f"<div class='result-box'>{output['answer']}</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        with st.expander("🔍 View Pipeline Details"):
            try:
                st.json(json.loads(output.get("messages", "[]")))
            except json.JSONDecodeError:
                st.text(output.get("messages", ""))
    else:
        st.error(result["error"])


# ───────────────────────────── Voice Form ──────────────────────────────────────
st.markdown(
    "### 🎤 Voice or Text Input\n\n"
    "Upload an audio file to use speech recognition, or type a command directly."
)

with st.form("voice_form"):
    audio_file = st.file_uploader(
        "🎙️ Upload Audio File",
        type=["wav", "mp3", "ogg", "flac"],
        help="Whisper supports WAV, MP3, OGG, FLAC. Larger files take longer to process.",
    )

    st.markdown("**— or —**")

    text_command = st.text_input(
        "⌨️ Type a text command (skips transcription)",
        placeholder="e.g., Explain what a transformer model is in simple terms",
    )

    submitted = st.form_submit_button("🎙️ Process", use_container_width=True)

if submitted:
    if audio_file is not None:
        # Encode audio to base64 for JSON transport
        audio_bytes = audio_file.read()
        audio_b64   = base64.b64encode(audio_bytes).decode("utf-8")

        st.info(
            f"Processing audio: {audio_file.name} "
            f"({len(audio_bytes) / 1024:.1f} KB)"
        )
        with st.spinner("Transcribing audio and generating response..."):
            result = call_model(audio_base64=audio_b64)
        render_result(result)

    elif text_command.strip():
        with st.spinner("Processing text command..."):
            result = call_model(question=text_command.strip())
        render_result(result)

    else:
        st.warning("Please upload an audio file or type a text command.")
