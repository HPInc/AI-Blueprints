import streamlit as st
import requests
import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Vacation Recommendation Agent", page_icon="🌍", layout="centered"
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
""",
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown(
    "<h1 style='text-align: center; color: #2C3E50;'>🏖️ Vacation Recommendation Agent 🌍</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h3 style='text-align: center; color: #555;'>Find the perfect vacation based on your preferences.</h3>",
    unsafe_allow_html=True,
)


# ─── Sidebar Instructions ────────────────────────────────────────────────────
with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    1. The model endpoint is automatically configured for deployment.

    2. Type or paste the type of vacation you would like to be recommended.

    3. Click **Get Recommendations** to see the result.
    """)

# ─── MLflow Endpoint Configuration ───────────────────────────────────────────
MLFLOW_ENDPOINT = "http://localhost:5002/invocations"

# --- User Input ---
query = st.text_input(
    "Enter your vacation preferences:",
    placeholder="e.g., Beach resort, adventure trip, budget vacation 🌴",
)


# --- Submit Button ---
if st.button("🔍 Get Recommendations"):
    if not query:
        st.warning("⚠️ Please enter a vacation preference!")
    else:
        # API Configuration
        api_url = MLFLOW_ENDPOINT
        payload = {"inputs": {"query": [query]}, "params": {"show_score": True}}

        # --- Loading Spinner ---
        with st.spinner("Fetching recommendations..."):
            try:
                response = requests.post(api_url, json=payload, verify=False)
                response.raise_for_status()
                data = response.json()

                # --- Display Results ---
                if "predictions" in data:
                    st.success("✅ Here are your vacation recommendations!")
                    for item in data["predictions"]:
                        st.markdown(
                            f"""
                            <div style="
                                background-color: #ffffff;
                                padding: 15px;
                                border-radius: 10px;
                                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                                margin: 10px 0px;
                                border-left: 8px solid #4CAF50;
                            ">
                                <h4 style="color: #2C3E50;">🏝️ {item['Pledge']}</h4>
                                <p><strong>Similarity Score:</strong> <span style="color: #4CAF50;">{item['Similarity']:.4f}</span></p>
                            </div>
                        """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.error("❌ Unexpected response format. Please try again.")

            except requests.exceptions.RequestException as e:
                st.error(
                    "❌ Error fetching recommendations. Please check your connection."
                )
                st.error(str(e))

st.warning(
    "Disclaimer: This application is provided for demonstration and illustrative purposes only. "
    "It does not represent a fully optimized or production-grade solution. "
    "Outputs may not be accurate, complete, or suitable for real-world decision-making. "
    "Results can often be improved by modifying the underlying code, models, data sources, and configuration."
)

st.write("Built with ❤️ using HP AI Studio")
