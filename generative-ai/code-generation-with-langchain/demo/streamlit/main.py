import streamlit as st
import os
import base64
import requests
from io import BytesIO
import numpy as np
from pathlib import Path
import json 


os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Code Generation with Langchain",
    page_icon = "🖥️",
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
""", unsafe_allow_html=True)

# --- Logo ---

def uri_from(path: Path) -> str:
    return f"data:image/{path.suffix[1:].lower()};base64," + base64.b64encode(path.read_bytes()).decode()

assets = Path("assets")
hp_uri = uri_from(assets / "HP-Logo.png")
ais_uri = uri_from(assets / "AI-Studio.png")
zhp_uri = uri_from(assets / "Z-HP-logo.png")

st.markdown(f"""
    <div style="display:flex;justify-content:space-between;
                align-items:center;margin-bottom:1.5rem">
        <img src="{hp_uri}"  alt="HP Logo" style="width:90px;height:auto;">
        <img src="{ais_uri}" alt="AI Studio Logo" style="width:90px;height:auto;">
        <img src="{zhp_uri}" alt="Z by HP Logo" style="width:90px;height:auto;">
    </div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Header 
# ─────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🧑‍💻 Code Generation with Langchain</h1>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 1 ▸ MLflow API Configuration
# ─────────────────────────────────────────────────────────────
# Standardized MLflow endpoint for containerized deployment
MLFLOW_ENDPOINT = "http://localhost:5002/invocations"
api_url = MLFLOW_ENDPOINT

    
# ─────────────────────────────────────────────────────────────
# 2 ▸ Main – data input
# ─────────────────────────────────────────────────────────────

user_question = st.text_input("Enter your query:")

    
# ─────────────────────────────────────────────────────────────
# 3 ▸ Call the model
# ─────────────────────────────────────────────────────────────

if st.button("🖥️ Get generated code"):
    if not user_question:
        st.warning("⚠️ Please enter your query")
    else:
        with st.spinner("Generating..."):
            payload = {
                "inputs": {"question": [user_question]},
            }
            try:
                response = requests.post(MLFLOW_ENDPOINT, json=payload, verify=False)
                response.raise_for_status()
                data = response.json()

                # Extract the 'result' key from the response
                result = data.get("result", "")
                
                # Pretty-print the result if it's JSON
                try:
                    parsed_result = json.loads(result)
                    pretty_result = json.dumps(parsed_result, indent=4)
                except (json.JSONDecodeError, TypeError):
                    pretty_result = str(result)

                if pretty_result:
                    st.success("✅ Here is your generated code!")

                    # Display the pretty JSON in a code block
                    st.markdown("""
                        <style>
                            .custom-code-box {
                                max-width: 800px;
                                margin: auto;
                                overflow-x: auto;
                            }
                        </style>
                    """, unsafe_allow_html=True)

                    st.markdown('<div class="custom-code-box">', unsafe_allow_html=True)
                    st.code(pretty_result, language='json')
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Prepare download
                    code_bytes = pretty_result.encode("utf-8")
                    b64 = base64.b64encode(code_bytes).decode()
                    href = f'<a href="data:file/txt;base64,{b64}" download="generated_code.json">📥 Download Code</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    st.error("❌ No code returned. Please try again.")
            except requests.exceptions.RequestException as e:
                st.error("❌ Error fetching code.")
                st.error(str(e))


# ─────────────────────────────────────────────────────────────
# 4 ▸ Footer
# ─────────────────────────────────────────────────────────────
st.markdown(
"""
> Built with ❤️ using HP AI Studio.
""",
unsafe_allow_html=True,
)
