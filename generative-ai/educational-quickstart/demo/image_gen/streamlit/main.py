"""
AI Learning Playground — Image Generator Demo

Focused Streamlit frontend for the AIStudio-EQ-ImageGen model.
This app sends requests to the registered ImageGenModel via the MLflow invocations endpoint.

Input sent to model:
    prompt — Text description of the image to generate

Output received:
    answer — Base64-encoded PNG image (decoded and displayed)

To start the MLflow model server:
    mlflow models serve -m models:/AIStudio-EQ-ImageGen/1 -p 5002 --no-conda

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
    page_title="AI Image Generator",
    page_icon="🎨",
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
    "<h2>🎨 AI Image Generator</h2>"
    "<p>Text-to-image generation powered by SDXL-Turbo — high quality in just 4 denoising steps</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ───────────────────────────── Sidebar ─────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown(
    """
**MLflow Server**

Start the image gen model server before using this app:

```bash
mlflow models serve \\
  -m models:/AIStudio-EQ-ImageGen/1 \\
  -p 5002 --no-conda
```

**About SDXL-Turbo**

SDXL-Turbo is a distilled diffusion model that generates
high-quality images in 1-4 steps. Guidance scale is set
to 0.0 (turbo mode — no classifier-free guidance).
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


def call_model(prompt: str, timeout: int = 600) -> dict:
    """
    Send a POST request to the ImageGenModel's invocations endpoint.

    Payload schema — only the column ImageGenModel expects:
        prompt (str)
    """
    payload = {
        "inputs": [{"prompt": prompt}],
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
                "mlflow models serve -m models:/AIStudio-EQ-ImageGen/1 -p 5002 --no-conda\n"
                "```"
            ),
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out — image generation can take 30–60s on first run.",
        }
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}


# ───────────────────────────── Image Form ──────────────────────────────────────
st.markdown(
    "### 🖼️ Generate an Image\n\n"
    "Describe any scene, object, or concept in natural language. "
    "Be specific — more detail leads to better images."
)

with st.form("image_form"):
    prompt = st.text_area(
        "✍️ Image Prompt",
        height=120,
        placeholder=(
            "e.g., A futuristic robot teaching in a university lecture hall, "
            "photorealistic, warm golden lighting, high detail\n\n"
            "e.g., Abstract visualization of a neural network, glowing blue nodes, "
            "dark background, digital art style"
        ),
    )
    submitted = st.form_submit_button("🎨 Generate Image", use_container_width=True)

if submitted:
    if not prompt.strip():
        st.warning("Please enter an image prompt.")
    else:
        with st.spinner("Generating image (this may take 30–60 seconds)..."):
            result = call_model(prompt)

        if result["success"]:
            answer = result["data"]["answer"]

            # The model returns either a base64-encoded PNG or an error message string.
            # Base64 PNG strings are typically >200 chars and don't start with ❌.
            if len(answer) > 200 and not answer.startswith("❌"):
                try:
                    img_bytes = base64.b64decode(answer)
                    st.markdown("### 🖼️ Generated Image")
                    st.image(
                        img_bytes,
                        caption=f'"{prompt[:80]}..."',
                        use_container_width=True,
                    )
                    st.divider()
                    with st.expander("📋 Request Details"):
                        st.json(json.loads(result["data"].get("messages", "[]")))
                except Exception as decode_err:
                    st.error(f"Failed to decode image: {decode_err}")
                    st.code(answer[:200])
            else:
                # Error message from the model
                st.markdown(
                    f"<div class='result-box'>{answer}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.error(result["error"])
