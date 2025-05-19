import streamlit as st
import os
import requests

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Handwritten Digit Classification",
    page_icon = "1️⃣",
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
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>✍️ Handwritten Digit Classification</h1>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 1 ▸ Server Settings
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model API Settings")

api_url = st.sidebar.text_input(
    "MLflow /invocations URL",
    value="https://localhost:5000/invocations",
    help="Endpoint where the MLflow model is served."
)

    
# ─────────────────────────────────────────────────────────────
# 2 ▸ Main – data input
# ─────────────────────────────────────────────────────────────
digit_image = st.file_uploader(
    "Choose a image:",
     type = ["jpg", "jpeg", "png"]
)


# ─────────────────────────────────────────────────────────────
# 3 ▸ Call the model
# ─────────────────────────────────────────────────────────────
if st.button("🖊️ Get Classification"):
    if not digit_image:
        st.warning("⚠️ Please enter a image!")
    else:
        file = {"files":digit_image}
        # --- Loading Spinner ---
        with st.spinner("Fetching recommendations..."):
            payload = {
                "inputs": {"digit_image": [digit_image]},
            }
            try:
                response = requests.post(api_url, payload, verify=False)
                response.raise_for_status()
                data = response()

                # --- Display Results ---
                if "predictions" in data:
                        st.success("✅ Here are your vacation recommendations!")
                        st.markdown(f"""
                            <div style="
                                background-color: #ffffff;
                                padding: 15px;
                                border-radius: 10px;
                                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                                margin: 10px 0px;
                                border-left: 8px solid #4CAF50;
                            ">
                                <h4 style="color: #2C3E50;">{data}</h4>
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
*✍️1️⃣Handwritten Digit Classification © 2025* local, private, handwritten classification + MLflow.

---
> Built with ❤️ using [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).
""",
unsafe_allow_html=True,
)