import streamlit as st
import pandas as pd
import requests, urllib3, base64
import altair as alt
from requests.exceptions import ConnectionError, HTTPError, Timeout
from pathlib import Path
import streamlit.components.v1 as components

# ────────────────  SET-UP  ─────────────────
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="Two Cities Forecast", page_icon="⏳", layout="centered")

# ────────────────  ALL-STYLES IN ONE BLOCK  ─────────────
st.markdown(
    """
    <style>
    /* ─── GLOBAL TYPOGRAPHY ───────────────────── */
    .stApp, .stApp *:not(h1) {
      font-family: "Segoe UI", sans-serif;
      font-size: 20px !important;
      line-height: 1.45 !important;
      color: #1E1E1E;
    }

    /* ─── PAGE BACKGROUND ─────────────────────── */
    .stApp {
      background: linear-gradient(180deg,
        #caedff 0%, #bde9fe 35%,
        #ace3fd 70%, #99ddfc 100%);
    }

    /* ─── CARD CONTAINER ─────────────────────── */
    .block-container {
      background: #f4fbff !important;
      width: 75vw !important; max-width: none !important;
      padding: 3rem 2.5rem !important;
      border-radius: 24px;
      box-shadow: 0 14px 48px rgba(0,97,255,0.12);
      margin: 3rem auto;
    }

    /* ─── TITLE ──────────────────────────────── */
    h1 {
      color: #004FCB;
      text-align: center;
      font-weight: 800;
      font-size: 2.8rem;
      margin-bottom: 0.75rem;
      letter-spacing: 0.02rem;
    }

    /* ─── BUTTON ─────────────────────────────── */
    .stButton > button,
    .stButton > button span {
      background-color: #004FCB;
      color: #FFFFFF !important;
      border: none;
      border-radius: 10px;
      padding: 0.85rem 1.9rem;
      font-size: 20px !important;
      font-weight: 600;
      transition: background 0.25s ease;
    }
    .stButton > button:hover,
    .stButton > button:focus {
      background: #2E7AFF !important;
    }
    /* make everything inside the button white */
    .stButton > button * {
      color: #FFFFFF !important;
    }

    /* ─── FORM ITEM LABELS ───────────────────── */
    div[data-baseweb="form-item"] > label {
      font-size: 20px !important;
      font-family: "Segoe UI", sans-serif;
    }

    /* ─── TEXT & NUMBER INPUTS ───────────────── */
    div[data-baseweb="input"] > div > input {
      height: 64px !important;    
      font-size: 20px !important;  
     padding: 0 1rem !important;  
    }
    
    
    /* enlarge the actual input element to fill the wrapper */
    div[data-baseweb="input"] > div > input {
      height: 100% !important;
    }
    /* plus/minus icons bigger */
    div[data-baseweb="input"] svg {
      width: 1.2rem; height: 1.2rem;
    }

    /* ─── SELECT DROPDOWNS ───────────────────── */
    div[data-baseweb="select"] > div {
      min-height: 64px !important;
      padding: 1rem 1rem !important;
      font-size: 20px !important;
      display: flex; align-items: center;
    }

    /* ─── ALERTS ─────────────────────────────── */
    .stAlert > div {
      background-color: #E5F1FF;
      border-left: 6px solid #004FCB;
      font-size: 20px;
      font-family: "Segoe UI", sans-serif;
    }

    /* ─── DIVIDER & FOOTER ───────────────────── */
    hr, .stHorizontalRule { border-color: rgba(0,77,204,0.20); }

    /* ─── LOGOS ──────────────────────────────── */
    img[alt="HP Logo"],
    img[alt="Z by HP Logo"],
    img[alt="AI Studio Logo"] {
      height: 90px;
    }

    /* ─── TABLE (scrollable) ─────────────────── */
    .forecast-table-container {
      max-height: 400px;
      overflow: auto;
      border: 1px solid #ddd;
    }
    .forecast-table-container table {
      border-collapse: collapse;
      width: 100%;
      font-family: "Segoe UI", sans-serif;
      font-size: 18px;
    }
    .forecast-table-container th,
    .forecast-table-container td {
      border: 1px solid #ddd;
      padding: 0.5rem;
      color: #929193;
    }
    .forecast-table-container th {
      font-size: 17px !important;;
    }
    .forecast-table-container td {
      font-size: 20px !important;;
    }

    /* ─── CHART SCALE TEXT ───────────────────── */
    .vega-embed .axis text,
    .vega-embed .axis-title,
    .vega-embed .legend-label,
    .vega-embed .legend-title {
      font-size: 24px !important;
      font-family: "Segoe UI", sans-serif;
    }
    /* catch all Vega-Lite text */
    .vega-embed text {
      font-size: 24px !important;
      font-family: "Segoe UI", sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ────────────────  LOGO ROW ─────────────────
def uri_from(path: Path) -> str:
    return f"data:image/{path.suffix[1:].lower()};base64," + base64.b64encode(path.read_bytes()).decode()

assets = Path(__file__).resolve().parent / "assets"
hp_uri = uri_from(assets / "HP-Logo.png")
z_uri  = uri_from(assets / "Z-HP-logo.png")
ais_uri= uri_from(assets / "AI-Studio.png")

st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1.2rem">
      <img src="{hp_uri}"  alt="HP Logo">
      <img src="{ais_uri}" alt="AI Studio Logo">
      <img src="{z_uri}"   alt="Z by HP Logo">
    </div>
""", unsafe_allow_html=True)

# ────────────────  MAIN UI & INFERENCE ────────
st.title("⏳ Two Cities Time Series Forecast")
st.write("Select a city and forecast horizon to see mobility + case forecasts.")

endpoint = st.text_input(
    "🔗 Model endpoint URL",
    value="https://localhost:52673/invocations",
    help="e.g. https://localhost:52673/invocations"
)
city  = st.selectbox("Select City", ["New York", "London"])
steps = st.number_input("Forecast horizon (days)", 1, 90, 14, step=1)

if st.button("Run Forecast"):
    payload = {"inputs": {"city": [city], "steps": [steps]}, "params": {}}
    hdrs = {"Accept": "application/json", "Content-Type": "application/json"}

    try:
        with st.spinner("🔄 Calling inference endpoint…"):
            r = requests.post(endpoint, json=payload, headers=hdrs, timeout=15, verify=False)
            r.raise_for_status()
            data = r.json()

        preds = data.get("predictions")
        df = pd.DataFrame(preds).round(2)
        st.success(f"Forecast for {city} — next {steps} days")

        # Table
        html = df.style.to_html()
        st.markdown(f'<div class="forecast-table-container">{html}</div>', unsafe_allow_html=True)

        # Charts (Altair with custom font sizes)
        df_reset = df.reset_index().rename(columns={"index":"day"})
        for col in df.columns:
            nice = col.replace("_forecast", "").replace("_", " ").title()
            st.subheader(f"{nice} Forecast")

            chart = (
                alt.Chart(df_reset)
                   .mark_line()
                   .encode(
                     x=alt.X("day:Q", axis=alt.Axis(labelFontSize=20, titleFontSize=20)),
                     y=alt.Y(f"{col}:Q", axis=alt.Axis(labelFontSize=20, titleFontSize=20))
                   )
                   .properties(height=300)
                   .configure_legend(labelFontSize=20, titleFontSize=20)
            )
            st.altair_chart(chart, use_container_width=True)

    except (ConnectionError, Timeout):
        st.error("🚫 Could not reach endpoint.")
    except HTTPError as he:
        st.error(f"🚨 HTTP {he.response.status_code}: {he.response.text}")
    except Exception as e:
        st.error(f"❗ Unexpected error: {e}")

st.write("---")
st.write("Built with ❤️ using HP AI Studio")
