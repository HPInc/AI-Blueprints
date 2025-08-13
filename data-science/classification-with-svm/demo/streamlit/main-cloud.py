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


# ────────────────  LOGO ROW  ─────────────────  
def uri_from(path: Path) -> str:
    return f"data:image/{path.suffix[1:].lower()};base64," + base64.b64encode(path.read_bytes()).decode()

assets = Path('data-science/classification-with-svm/demo/streamlit') / "assets"
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

# MLflow endpoint configured for deployment
MLFLOW_ENDPOINT = "https://ffe8d707c6a1.ngrok.app/invocations"

# Feature sliders
st.header("Input Features (cm)")
sepal_length = st.slider("Sepal length", 4.0, 10.0, 5.4, 0.1)
sepal_width  = st.slider("Sepal width",  2.0, 10.0, 3.4, 0.1)
petal_length = st.slider("Petal length", 1.0, 10.0, 1.3, 0.1)
petal_width  = st.slider("Petal width",  0.1, 10.0, 0.2, 0.1)

# ────────────  PREDICTION LOGIC  ─────────────  
if st.button("Predict Species"):
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
            resp = requests.post(MLFLOW_ENDPOINT, json=payload, headers=headers,
                                  timeout=10, verify=False)
            resp.raise_for_status()
            data = resp.json()

        preds = data.get("predictions")
        if not isinstance(preds, list) or not preds:
            st.error("⚠️ Endpoint returned no predictions.")
        else:
            st.success(f"✅ Predicted Iris Species: **{preds[0]}**")

    except ConnectionError:
        st.error(f"🚫 Could not connect to `{MLFLOW_ENDPOINT}`. Is the server running?")
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
