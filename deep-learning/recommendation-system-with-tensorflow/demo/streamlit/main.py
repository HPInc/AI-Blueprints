import streamlit as st
import os
import requests
from pathlib import Path
import base64

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Movie Recommendation Agent",
    page_icon = "🎬",
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
                align-items:center;margin-bottom:1.5rem;
                background-color: transparent;
            ">
        <img src="{hp_uri}"  alt="HP Logo" style="width:90px;height:auto;">
        <img src="{ais_uri}" alt="AI Studio Logo" style="width:90px;height:auto;">
        <img src="{zhp_uri}" alt="Z by HP Logo" style="width:90px;height:auto;">
    </div>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🎥 Movie Recommendation Agent</h1>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: #555;'> Have a movie recommendation based on your movie rating.</h3>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 1 ▸ MLflow API Configuration
# ─────────────────────────────────────────────────────────────
# Standardized MLflow endpoint for containerized deployment
MLFLOW_ENDPOINT = "http://localhost:5002/invocations"
api_url = MLFLOW_ENDPOINT

    
# ─────────────────────────────────────────────────────────────
# 2 ▸ Main – data input
# ─────────────────────────────────────────────────────────────
movie_id = st.number_input(
    "Enter a movie ID:",
     min_value = 0
)
rating = st.number_input(
    "Enter a rating", 
    min_value = 0.0,
    max_value = 5.0,
    step=0.5
)

# ─────────────────────────────────────────────────────────────
# 3 ▸ Call the model
# ─────────────────────────────────────────────────────────────
if st.button("🍿 Get Recommendations"):
    if not movie_id:
        st.warning("⚠️ Please enter a Movie ID!")
    elif not rating:
        st.warning("⚠️ Please enter a rating!")
    else:
        # --- Loading Spinner ---
        with st.spinner("Fetching recommendations..."):
            payload = {
                "inputs": {"movie_id": [movie_id], "rating":[rating]},
            }
            try:
                response = requests.post(api_url, json=payload, verify=False)
                response.raise_for_status()
                data = response.json()
                
                # --- Display Results ---
                if "predictions" in data:
                    st.success("✅ Here are your movie recommendations!")
                    for i, movie in enumerate(data['predictions'], 1):
                        title = movie[0]
                        score = movie[1]
                        st.markdown(f"""
                            <div style="
                                background-color: #ffffff;
                                padding: 15px;
                                border-radius: 10px;
                                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                                margin: 10px 0px;
                                border-left: 8px solid #4CAF50;
                            ">
                                <h4 style="color: #2C3E50;">🍿{title}</h4>
                                <p><strong>Score:</strong> <span style="color: #4CAF50;">{score:.4f}</span></p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("❌ Unexpected response format. Please try again.")

            except requests.exceptions.RequestException as e:
                st.error("❌ Error fetching recommendations.")
                st.error(str(e))
# ─────────────────────────────────────────────────────────────
# 4 ▸ Footer
# ─────────────────────────────────────────────────────────────
st.markdown(
"""
*🎥🍿Recommender Movies System © 2025* local, private, recommender system + MLflow.

---
> Built with ❤️ using [**HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).
""",
unsafe_allow_html=True,
)
