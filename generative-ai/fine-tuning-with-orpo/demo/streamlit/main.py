import streamlit as st
import base64
import requests
import json
from pathlib import Path

st.set_page_config(page_title="🤖 Fine Tuning with Orpo", layout="centered")

st.markdown("""
    <style>
        .result-box {
            border: 2px solid #4e54c8;
            border-radius: 10px;
            padding: 1rem;
            margin-top: 1rem;
            background-color: #f4f6ff;
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

st.title("Fine Tuning with Orpo")

# ─────────────────────────────────────────────────────────────
# 1 ▸ MLflow API Configuration
# ─────────────────────────────────────────────────────────────
# Standardized MLflow endpoint for containerized deployment

MLFLOW_ENDPOINT = "http://localhost:5002/invocations"
api_url = MLFLOW_ENDPOINT

# ─────────────────────────────────────────────────────────────
# 2 ▸ Main – data input
# ─────────────────────────────────────────────────────────────

user_prompt = st.text_input("Add a Prompt")

user_finetuning = st.checkbox("Use Finetuning")

max_tokens = st.number_input("Tokens", value=0, step=1)

# ─────────────────────────────────────────────────────────────
# 3 ▸ Call the model
# ─────────────────────────────────────────────────────────────

if st.button("Get response"):
    if not user_prompt or not max_tokens:
        st.warning("⚠️ Please fill all the fields")
    else:
        with st.spinner("Generating response..."):
            payload = {
                "inputs":{
                    "prompt":[user_prompt],
                    "use_finetuning":[user_finetuning],
                    "max_tokens":[max_tokens]
                }
            }

           
        try:
            response = requests.post(api_url, json=payload, verify=False)
            response.raise_for_status()
            data = response.json()
            
            # Extract the actual string from the list of dictionaries
            raw_response = data.get("predictions")[0].get("response")
            
            # Replace newline characters with <br>
            formatted_response = raw_response.replace("\n", "<br>")
            
            if formatted_response:
                st.success("✅ Here is your response!")
                st.markdown(f'<div class="result-box">{formatted_response}</div>', unsafe_allow_html=True)
            else:
                st.error("❌ Unexpected response format. Please try again.")
        except requests.exceptions.RequestException as e:
            st.error("❌ Error fetching response.")
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