import streamlit as st
import requests, urllib3, base64
from requests.exceptions import ConnectionError, HTTPError, Timeout
from pathlib import Path

# ────────────────  SET-UP  ─────────────────  
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# ────────────────  ALL-STYLES IN ONE BLOCK  ─────────────  
st.markdown(
    """
    <style>
    /* GLOBAL TYPOGRAPHY */
    .stApp, .stApp *:not(h1) {
      font-family: "Segoe UI", sans-serif;
      font-size: 20px !important;
      line-height: 1.45 !important;
      color: #1E1E1E;
    }
    /* PAGE BACKGROUND */
    .stApp {
      background: linear-gradient(180deg,
        #fcdedc 0%, #ffc8c4 35%,
        #ffada8 70%, #ff918a 100%);
    }
    /* CARD CONTAINER */
    .block-container {
      background: #FFFFFF !important;
      width: 75vw !important; max-width: none !important;
      padding: 3rem 2rem !important;
      border-radius: 18px;
      box-shadow: 0 8px 25px rgba(0,0,0,0.08);
      margin: 2rem auto;
    }
    /* TITLE */
    h1 {
      color: #fa564b;
      text-align: center;
      font-weight: 800;
      font-size: 2.8rem;
      margin-bottom: 0.5rem;
    }
    /* BUTTON */
    .stButton > button,
    .stButton > button span {
      background-color: #fa372a;
      color: #FFFFFF !important;
      border: none;
      border-radius: 8px;
      padding: 0.8rem 1.6rem;
      font-size: 20px !important;
      transition: background 0.3s;
    }
    .stButton > button:hover {
      background-color: #5063a6 !important;
    }
    /* ensure button text stays white */
    .stButton > button * {
      color: #FFFFFF !important;
    }
    /* ALERTS */
    .stAlert > div {
      background-color: #e8f0fe;
      border-left: 6px solid #334f8d;
      font-size: 20px;
    }
    /* FORM LABELS */
    div[data-baseweb="form-item"] > label {
      font-size: 20px !important;
    }
    /* TEXT & NUMBER INPUTS */
    div[data-baseweb="input"] > div > input {
      height: 56px !important;
      font-size: 20px !important;
      padding: 0 1rem !important;
    }
    div[data-baseweb="input"] svg {
      width: 1.2rem; height: 1.2rem;
    }
    /* SLIDERS */
    .stSlider > div input {
      height: 56px !important;
    }
    /* DIVIDER & FOOTER */
    hr, .stHorizontalRule { border-color: rgba(0,77,204,0.20); }
    /* LOGOS */
    img[alt="HP Logo"],
    img[alt="AI Studio Logo"],
    img[alt="Z by HP Logo"] {
      height: 90px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ────────────────  LOGO ROW  ─────────────────  
def uri_from(path: Path) -> str:
    return f"data:image/{path.suffix[1:].lower()};base64," + base64.b64encode(path.read_bytes()).decode()

assets = Path(__file__).parent / "assets"
hp_uri  = uri_from(assets / "HP-Logo.png")
ais_uri = uri_from(assets / "AI-Studio.png")
zhp_uri = uri_from(assets / "Z-HP-logo.png")

st.markdown(f"""
  <div style="display:flex;justify-content:space-between;
              align-items:center;margin-bottom:1.5rem">
    <img src="{hp_uri}"  alt="HP Logo">
    <img src="{ais_uri}" alt="AI Studio Logo">
    <img src="{zhp_uri}" alt="Z by HP Logo">
  </div>
""", unsafe_allow_html=True)

# ────────────────  CORE UI ──────────────────  
st.title("🌸 Iris Flower Classifier")
st.write("Provide flower measurements and get the predicted Iris species.")

# Endpoint URL input
endpoint = st.text_input(
    "🔗 Enter your model endpoint URL",
    value="https://localhost:52656/invocations",
    placeholder="e.g. https://localhost:52656/invocations"
)

# Feature sliders
st.header("Input Features (cm)")
sepal_length = st.slider("Sepal length", 4.0, 10.0, 5.4, 0.1)
sepal_width  = st.slider("Sepal width",  2.0, 10.0, 3.4, 0.1)
petal_length = st.slider("Petal length", 1.0, 10.0, 1.3, 0.1)
petal_width  = st.slider("Petal width",  0.1, 10.0, 0.2, 0.1)

# ────────────  PREDICTION LOGIC  ─────────────  
if st.button("Predict Species"):
    if not endpoint.lower().startswith(("http://","https://")):
        st.error("🚫 Please enter a valid URL starting with `http://` or `https://`.")
    else:
        payload = {
            "inputs": {
                "sepal-length": [sepal_length],
                "sepal-width":  [sepal_width],
                "petal-length": [petal_length],
                "petal-width":  [petal_width],
            },
            "params": {}
        }
        headers = {"Accept":"application/json","Content-Type":"application/json"}

        try:
            with st.spinner("🔄 Calling inference endpoint…"):
                resp = requests.post(endpoint, json=payload, headers=headers,
                                      timeout=10, verify=False)
                resp.raise_for_status()
                data = resp.json()

            preds = data.get("predictions")
            if not isinstance(preds, list) or not preds:
                st.error("⚠️ Endpoint returned no predictions.")
            else:
                st.success(f"✅ Predicted Iris Species: **{preds[0]}**")

        except ConnectionError:
            st.error(f"🚫 Could not connect to `{endpoint}`. Is the server running?")
        except Timeout:
            st.error("⏰ The request timed out. Try again or increase timeout.")
        except HTTPError as he:
            st.error(f"🚨 HTTP {he.response.status_code}: {he.response.text}")
        except ValueError:
            st.error("❓ Received invalid JSON from the endpoint.")
        except Exception as e:
            st.error(f"❗ An unexpected error occurred: {e}")

# ────────────────  FOOTER  ─────────────────  
st.write("---")
st.write("Built with ❤️ using HP AI Studio")
