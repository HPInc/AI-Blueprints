import streamlit as st
import base64
import requests
import json
from pathlib import Path

st.set_page_config(page_title="🤖 Fine Tuning with Orpo", layout="centered")

st.markdown("""
    <style>
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

st.markdown("""
<div style='background-color:#fff3cd;padding:1rem;border-left:6px solid #ffeeba;'>
<strong>Note:</strong> The UI works smoothly only up to <strong>1000 lines of code</strong> due to API timeout.
For larger inputs, refer the README.
</div>
""", unsafe_allow_html=True)

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

user_finetuning = st.checkbox("Finetuning")

max_tokens = st.number_input("Tokens", value=0, step=1)

# ─────────────────────────────────────────────────────────────
# 3 ▸ Call the model
# ─────────────────────────────────────────────────────────────

generate = st.button("Generate response")

if generate:
    st.session_state.test_script = ""

    if not user_prompt or user_finetuning or max_tokens:
        st.error("Please fill all the fields.")
    else:
        with st.spinner("Generating response..."):
            try:
                payload = {
                    "inputs":{
                        "prompt":[user_prompt],
                        "use_finetuning":[user_finetuning],
                        "max_tokens":[max_tokens]
                    }
                }

                response = requests.post(
                    api_url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    verify=False
                )
                response.raise_for_status()
                resp = response.json().get("predictions", response.json())

                st.session_state.test_script = resp.get("output_file_string", "[No script content returned]")
                st.session_state.script_b64 = resp.get("output_file_b64", "")

            except Exception as e:
                st.error(f"Error: {e}")
# --- Results ---
if "test_script" in st.session_state and st.session_state.test_script:
    st.subheader("🤖 Generated response")
    st.text_area("Test Script", value=st.session_state.test_script, height=300)

# ─────────────────────────────────────────────────────────────
# 4 ▸ Footer
# ─────────────────────────────────────────────────────────────
st.markdown(
"""
> Built with ❤️ using HP AI Studio.
""",
unsafe_allow_html=True,
)