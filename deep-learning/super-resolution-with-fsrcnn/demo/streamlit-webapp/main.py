import streamlit as st
import os
import base64
import requests
from io import BytesIO
from PIL import Image


os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Image Super Resolution",
    page_icon = "📷",
    layout="centered"
)

# --- Custom Styling ---
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

# ─────────────────────────────────────────────────────────────
# 1 ▸ Server Settings
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model API Settings")

api_url = st.sidebar.text_input(
    "MLflow /invocations URL",
    value="https://localhost:5000/invocations",
    help="Endpoint where the MLflow model is served."
)

    
# ─────────────────────────────────────────────────────────────
# 2 ▸ Main – data input
# ─────────────────────────────────────────────────────────────
digit_image = st.file_uploader(
    "Choose a image:",
     type = ["jpg", "jpeg", "png"]
)

if digit_image is not None:
    st.image(digit_image, width = 300)
    encoded_string = base64.b64encode(digit_image.read()).decode("utf-8")
else:
    st.text("Upload image")
    

# ─────────────────────────────────────────────────────────────
# 3 ▸ Call the model
# ─────────────────────────────────────────────────────────────
if st.button("Get the image with super resolution"):
    if not digit_image:
        st.warning("⚠️ Please enter a image!")
    else:
        file = {"files":digit_image}
        # --- Loading Spinner ---
        with st.spinner("Classifying..."):
            payload = {
                "inputs": {"image": [encoded_string]},
            }
            try:
                response = requests.post(api_url, json = payload, verify=False)
                response.raise_for_status()
                data = response.json()

                # --- Display Results ---
                if "predictions" in data:
                        
                        base64_image = data["predictions"]

                        image_bytes = base64.b64decode(base64_image)
                        image = Image.open(BytesIO(image_bytes))    
                        st.success("✅ Here are your image!")
                        st.image(image, caption="Super Resolution Output", use_column_width=True)

                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_bytes = buffered.getvalue()

                        st.download_button( 
                            label="📥 Download Image",
                            data=img_bytes,
                            file_name="super_resolution_output.png",
                            mime="image/png"
                            )

                else:
                    st.error("❌ Unexpected response format. Please try again.")

            except requests.exceptions.RequestException as e:
                st.error("❌ Error fetching classification.")
                st.error(str(e))
# ─────────────────────────────────────────────────────────────
# 4 ▸ Footer
# ─────────────────────────────────────────────────────────────
st.markdown(
"""
*🖼️ Image Super Resolution © 2025* local, private, handwritten classification + MLflow.

---
> Built with ❤️ using [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).
""",
unsafe_allow_html=True,
)