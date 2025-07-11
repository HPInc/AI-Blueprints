import streamlit as st
import os
import base64
import requests
from io import BytesIO
from PIL import Image

# Environment configuration
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

# --- Page Configuration ---
st.set_page_config(
    page_title="Image Super Resolution",
    page_icon="📷",
    layout="centered"
)

# --- Custom Style ---
st.markdown("""
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
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🖼️ Image Super Resolution</h1>", unsafe_allow_html=True)

# --- API Settings ---
st.sidebar.header("⚙️ Model API Settings")
api_url = st.sidebar.text_input(
    "MLflow /invocations URL",
    value="https://localhost:5000/invocations",
    help="Endpoint where the MLflow model is served."
)

# --- Image Upload ---
digit_image = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])

if digit_image is not None:
    st.session_state["uploaded_image"] = digit_image
    st.image(digit_image, width=300)
    encoded_string = base64.b64encode(digit_image.read()).decode("utf-8")
    st.session_state["encoded_image"] = encoded_string
else:
    st.text("Upload image")

# --- Button to Call the Model ---
if st.button("Get the image with super resolution"):
    if "encoded_image" not in st.session_state:
        st.warning("⚠️ Please upload an image!")
    else:
        payload = {
            "inputs": {"image": [st.session_state["encoded_image"]]},
        }

        with st.spinner("Enhancing image..."):
            try:
                response = requests.post(api_url, json=payload, verify=False)
                response.raise_for_status()
                data = response.json()

                base64_image = data.get("predictions", [None])[0]

                if base64_image and isinstance(base64_image, str):
                    image_bytes = base64.b64decode(base64_image)
                    image = Image.open(BytesIO(image_bytes))
                    st.session_state["enhanced_image"] = image
                    st.success("✅ Here is your image!")
                else:
                    st.error("❌ No valid image data returned from the model.")
                    st.write("Raw response:", data)

            except requests.exceptions.RequestException as e:
                st.error("❌ Error fetching prediction.")
                st.error(str(e))

# --- Display Enhanced Image and Download Button ---
if "enhanced_image" in st.session_state:
    st.image(st.session_state["enhanced_image"], caption="Super Resolution Output", use_container_width=True)

    buffer = BytesIO()
    st.session_state["enhanced_image"].save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label="📥 Download Image",
        data=buffer,
        file_name="super_resolution_output.png",
        mime="image/png"
    )

# --- Footer ---
st.markdown(
    """
    *🖼️ Image Super Resolution © 2025* local, private, super resolution + MLflow.

    ---
    > Built with ❤️ using **Z by HP AI Studio**.
    """,
    unsafe_allow_html=True,
)
