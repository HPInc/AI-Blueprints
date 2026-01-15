import streamlit as st
import os
import base64
import requests
from io import BytesIO
from PIL import Image
from pathlib import Path

# Environment configuration
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

# --- Page Configuration ---
st.set_page_config(
    page_title="Image Generation with Stable Diffusion XL",
    page_icon="📷",
    layout="centered",
)

# --- Custom Style ---
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 0 !important;
        }
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f4;
        }
        .stButton>button {
            background-color: #4CAF50 !important;
            color: white !important;
            font-size: 18px !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            border: none !important;
        }
        .stTextInput>div>div>input {
            font-size: 16px !important;
            padding: 10px !important;
        }
        .stMarkdown {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            margin: 10px 0px;
        }
        hr, .stHorizontalRule {
            border-color: rgba(0,77,204,0.20);
        }
        img[alt="HP Logo"],
        img[alt="AI Studio Logo"],
        img[alt="Z by HP Logo"] {
    width: 50px !important;
    height: auto !important;
}

    </style>
""",
    unsafe_allow_html=True,
)

# --- Logo ---


def uri_from(path: Path) -> str:
    return (
        f"data:image/{path.suffix[1:].lower()};base64,"
        + base64.b64encode(path.read_bytes()).decode()
    )


assets = Path("assets")
hp_uri = uri_from(assets / "HP-Logo.png")
ais_uri = uri_from(assets / "AI-Studio.png")
zhp_uri = uri_from(assets / "Z-HP-logo.png")

st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;
                align-items:center;margin-bottom:1.5rem">
        <img src="{hp_uri}"  alt="HP Logo" style="width:90px;height:auto;">
        <img src="{ais_uri}" alt="AI Studio Logo" style="width:90px;height:auto;">
        <img src="{zhp_uri}" alt="Z by HP Logo" style="width:90px;height:auto;">
    </div>
""",
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown(
    "<h1 style='text-align: center; color: #2C3E50;'>🖼️ Image Generation with Stable Diffusion XL</h1>",
    unsafe_allow_html=True,
)

# --- MLflow API Configuration ---
# Standardized MLflow endpoint for containerized deployment
MLFLOW_ENDPOINT = "http://localhost:5002/invocations"
api_url = MLFLOW_ENDPOINT

# --- data inputs ---

prompt = st.text_input("Prompt (image description):")
use_finetuning = st.checkbox("Use fine-tuning")

# SDXL native resolution is 1024x1024, minimum 768x768
st.info("ℹ️ SDXL works best with dimensions between 768-2048px. Native resolution is 1024x1024.")
height = st.number_input("Image height (px):", min_value=512, max_value=2048, value=1024, step=64)
width = st.number_input("Image width (px):", min_value=512, max_value=2048, value=1024, step=64)

# Validation warning for sub-optimal dimensions
if height < 768 or width < 768:
    st.warning("⚠️ Dimensions below 768px may produce lower quality results with SDXL. Consider using 1024x1024 for best results.")

num_images = st.number_input("Number of images:", min_value=1, max_value=10, value=1)
num_inference_steps = st.number_input(
    "Number of inference steps:", min_value=1, max_value=100, value=50
)


# --- Button to Call the Model ---
if st.button("Get result"):
    if not all([prompt, height, width, num_images, num_inference_steps]):
        st.warning("⚠️ Please fill all fields!")
    else:
        payload = {
            "inputs": {
                "prompt": [prompt],
                "use_finetuning": [use_finetuning],
                "height": [height],
                "width": [width],
                "num_images": [num_images],
                "num_inference_steps": [num_inference_steps],
            },
        }

        with st.spinner("Generating image..."):
            try:
                response = requests.post(api_url, json=payload, verify=False)
                response.raise_for_status()
                data = response.json()

                # ✅ Save predictions to session state
                st.session_state["predictions"] = data.get("predictions", [])

            except requests.exceptions.RequestException as e:
                st.error("❌ Error fetching prediction.")
                st.error(str(e))

# ✅ Always display images if they exist in session state
if "predictions" in st.session_state:
    predictions = st.session_state["predictions"]
    for i, pred in enumerate(predictions):
        base64_image = pred.get("output_images", "")
        if base64_image and isinstance(base64_image, str):
            try:
                image_bytes = base64.b64decode(base64_image)
                image = Image.open(BytesIO(image_bytes))
                st.image(image, caption=f"Image {i+1}", use_container_width=True)

                buffer = BytesIO()
                image.save(buffer, format="PNG")
                buffer.seek(0)

                st.download_button(
                    label=f"📥 Download Image {i+1}",
                    data=buffer,
                    file_name=f"image_{i+1}.png",
                    mime="image/png",
                )
            except Exception as e:
                st.error(f"❌ Error displaying image {i+1}: {str(e)}")
        else:
            st.warning(f"⚠️ No image data found for prediction {i+1}.")

# --- Footer ---
st.markdown(
    """
    > Built with ❤️ using HP AI Studio.
    """,
    unsafe_allow_html=True,
)
