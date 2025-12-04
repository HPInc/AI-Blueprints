import streamlit as st
import os
import requests
import base64
import json
from pathlib import Path


os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Agentic RAG with TensorRT-LLM", page_icon="🤖", layout="centered"
)

# --- Custom Styling ---
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
        .stTextArea>div>div>textarea {
            font-size: 16px !important;
            padding: 10px !important;
            border: 2px solid #e0e0e0 !important;
            border-radius: 8px !important;
        }
        .stTextArea>div>div>textarea:focus {
            border-color: #4CAF50 !important;
        }
        .stMarkdown {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0px;
        }
        hr, .stHorizontalRule {
            border-color: rgba(0,77,204,0.20);
        }
        img[alt="HP Logo"],
        img[alt="AI Studio Logo"],
        img[alt="Z by HP Logo"] {
            width: 70px !important;
            height: auto !important;
        }
        .info-box {
            background-color: #f0f7ff;
            padding: 18px;
            border-radius: 12px;
            border-left: 4px solid #2196F3;
            margin: 15px 0px;
        }
        .context-box {
            background-color: #fafafa;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            margin: 10px 0px;
            font-size: 14px;
            max-height: 300px;
            overflow-y: auto;
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
        <img src="{hp_uri}"  alt="HP Logo" style="width:90px;height:auto;">
        <img src="{ais_uri}" alt="AI Studio Logo" style="width:90px;height:auto;">
        <img src="{zhp_uri}" alt="Z by HP Logo" style="width:90px;height:auto;">
    </div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align: center; color: #2C3E50; background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>"
    "🤖 Agentic RAG with TensorRT-LLM</h1>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# 1 ▸ MLflow API Configuration
# ─────────────────────────────────────────────────────────────
# Standardized MLflow endpoint for containerized deployment
MLFLOW_ENDPOINT = "http://localhost:5002/invocations"
api_url = MLFLOW_ENDPOINT


def normalize_prediction_entry(entry):
    """Return a dict with answer/context fields regardless of backend output shape."""
    if isinstance(entry, dict):
        return entry

    if isinstance(entry, str):
        try:
            parsed = json.loads(entry)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        return {"answer": entry}

    if isinstance(entry, (list, tuple)):
        joined = "\n\n".join(str(item) for item in entry)
        return {"answer": joined}

    return {"answer": str(entry)}


# ─────────────────────────────────────────────────────────────
# 2 ▸ Main – Data Input
# ─────────────────────────────────────────────────────────────

# Example queries for user reference
st.markdown("### 💬 Ask a Question")
user_query = st.text_input(
    "Enter your question about AI Studio:", placeholder="Type your question here..."
)

# ─────────────────────────────────────────────────────────────
# 3 ▸ Call the Model
# ─────────────────────────────────────────────────────────────
if st.button("🔍 Get Answer"):
    if not user_query.strip():
        st.warning("⚠️ Please enter a question!")
    else:
        # --- Loading Spinner ---
        with st.spinner("Processing your query through the Agentic RAG system..."):
            payload = {"inputs": {"query": user_query.strip()}}

            try:
                response = requests.post(api_url, json=payload, verify=False)
                response.raise_for_status()
                data = response.json()

                # --- Display Results ---
                if "predictions" in data:
                    predictions_raw = data.get("predictions", [])

                    if isinstance(predictions_raw, dict):
                        predictions = [predictions_raw]
                    elif isinstance(predictions_raw, (str, int, float)):
                        predictions = [predictions_raw]
                    else:
                        predictions = predictions_raw

                    if not isinstance(predictions, list):
                        predictions = [predictions]

                    parsed_predictions = [
                        normalize_prediction_entry(item) for item in predictions
                    ]

                    if not parsed_predictions:
                        st.error("❌ No predictions returned from the model.")
                    else:
                        for result in parsed_predictions:
                            st.markdown(
                                f"""
                                <div style="
                                    background-color: #f9f9f9;
                                    padding: 24px;
                                    border-radius: 12px;
                                    margin: 15px 0px;
                                    border-left: 5px solid #4CAF50;
                                    border: 1px solid #e8e8e8;
                                ">
                                    <h4 style="color: #2C3E50; margin-bottom: 12px; font-weight: 600;">📝 Answer:</h4>
                                    <p style="color: #34495E; line-height: 1.8; margin: 0;">{result.get('answer', 'No answer available')}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            with st.expander("🔎 Raw Response Entry"):
                                st.json(result)
                else:
                    st.error("❌ Unexpected response format. Please try again.")
                    st.json(data)

            except requests.exceptions.RequestException as e:
                st.error("❌ Error connecting to the MLflow endpoint.")
                st.error(f"Details: {str(e)}")
                st.info(
                    """
                **Troubleshooting Tips:**
                - Ensure the MLflow model is deployed and running
                - Check that the endpoint URL is correct
                - Verify the model is registered in MLflow
                - Make sure the service is accessible on port 5002
                """
                )
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")


# ─────────────────────────────────────────────────────────────
# 5 ▸ Footer
# ─────────────────────────────────────────────────────────────
st.write("---")
st.write("Built with ❤️ using HP AI Studio")
