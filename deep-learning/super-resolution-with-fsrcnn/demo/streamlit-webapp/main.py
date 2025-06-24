import streamlit as st
import os
import base64
import requests
from io import BytesIO
from PIL import Image

# Configuração de ambiente
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

# --- Configuração da Página ---
st.set_page_config(
    page_title="Image Super Resolution",
    page_icon="📷",
    layout="centered"
)

# --- Estilo Personalizado ---
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

# --- Cabeçalho ---
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🖼️ Image Super Resolution</h1>", unsafe_allow_html=True)

# --- Configurações da API ---
st.sidebar.header("⚙️ Model API Settings")
api_url = st.sidebar.text_input(
    "MLflow /invocations URL",
    value="https://localhost:5000/invocations",
    help="Endpoint where the MLflow model is served."
)

# --- Upload da Imagem ---
digit_image = st.file_uploader(
    "Choose an image:",
    type=["jpg", "jpeg", "png"]
)

encoded_string = None
if digit_image is not None:
    st.image(digit_image, width=300)
    encoded_string = base64.b64encode(digit_image.read()).decode("utf-8")
else:
    st.text("Upload image")

# --- Botão para Chamar o Modelo ---
if st.button("Get the image with super resolution"):
    if not digit_image:
        st.warning("⚠️ Please upload an image!")
    else:
        payload = {
            "inputs": {"image": [encoded_string]},
        }

        with st.spinner("Enhancing image..."):
            try:
                response = requests.post(api_url, json=payload, verify=False)
                response.raise_for_status()
                data = response.json()

                # Corrigido: acessando a chave correta
                base64_image = data.get("predictions", [None])[0]

                if base64_image and isinstance(base64_image, str):
                    try:
                        image_bytes = base64.b64decode(base64_image)
                        image = Image.open(BytesIO(image_bytes))
                        st.success("✅ Here is your image!")
                        st.image(image, caption="Super Resolution Output", use_column_width=True)
                    except Exception as e:
                        st.error("❌ Failed to decode the image.")
                        st.error(str(e))
                else:
                    st.error("❌ No valid image data returned from the model.")
                    st.write("Raw response:", data)

            except requests.exceptions.RequestException as e:
                st.error("❌ Error fetching prediction.")
                st.error(str(e))

# --- Rodapé ---
st.markdown(
    """
    *🖼️ Image Super Resolution © 2025* local, private, handwritten classification + MLflow.

    ---
    > Built with ❤️ using **Z by HP AI Studio**.
    """,
    unsafe_allow_html=True,
)
