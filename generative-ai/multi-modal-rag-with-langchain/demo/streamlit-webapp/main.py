# app.py
import streamlit as st
import requests
import json
import pandas as pd
import warnings
from pathlib import Path
import ast

# Ignore SSL warnings for local development
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# --- Define project paths ---
# IMPORTANT: Make sure this path is correct for your project structure.
IMAGE_DIR = Path("../../data/context/images")

# --- Page Configuration & Custom CSS ---

st.set_page_config(
    page_title="ADO Wiki AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

def load_css():
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
        html, body, [class*="st-"] {
            font-family: 'Roboto', sans-serif;
        }

        /* Light mode styles */
        body {
            background-color: #FFFFFF;
            color: #262626;
        }

        .stApp {
            background-color: #FFFFFF;
        }

        /* Dark mode overrides */
        @media (prefers-color-scheme: dark) {
            body {
                background-color: #0e1117;
                color: #FFFFFF;
            }

            .stApp {
                background-color: #0e1117;
            }

            div[data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"]) {
                background-color: #1e1e1e !important;
                color: #f0f0f0 !important;
            }

            div[data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"] p) {
                background-color: #005c94 !important;
                color: #FFFFFF !important;
            }

            [data-testid="stSidebar"] {
                background-color: #1e1e1e !important;
            }

            [data-testid="stChatInput"] {
                background-color: #0e1117;
                border-top: 1px solid #333;
            }

            [data-testid="stChatInput"] textarea {
                background-color: #1e1e1e !important;
                color: #FFFFFF !important;
                border: 1px solid #444 !important;
            }

            [data-testid="stChatInput"] button {
                background-color: #0076a8 !important;
            }

            [data-testid="stChatInput"] button:hover {
                background-color: #005c94 !important;
            }
            
            .image-gallery-header {
                color: #cccccc;
                border-top: 1px solid #444;
            }

            /* Style for metric labels in dark mode */
            [data-testid="stMetricLabel"] {
                color: #a0a0a0 !important;
            }
        }

        /* Style for metric labels in light mode */
        [data-testid="stMetricLabel"] {
            color: #555555 !important;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# --- API Interaction Logic ---

def call_mlflow_api(api_url: str, query: str, force_regenerate: bool) -> dict:
    """Calls the MLflow model serving endpoint."""
    payload = {
        "dataframe_records": [
            {
                "query": query,
                "force_regenerate": force_regenerate,
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(api_url, json=payload, headers=headers, verify=False, timeout=120)
        
        if response.status_code == 200:
            predictions = response.json().get("predictions", [])
            if predictions:
                return {"success": True, "data": predictions[0]}
            else:
                return {"success": False, "error": "API returned an empty prediction."}
        else:
            error_message = f"API Error (Status {response.status_code}): {response.text}"
            return {"success": False, "error": error_message}
            
    except requests.exceptions.RequestException as e:
        error_message = f"Network or connection error: {e}"
        return {"success": False, "error": error_message}

# --- Main Application UI ---

def main():
    """Renders the main HP-branded Chatbot application page."""
    load_css()
    
    # --- Sidebar for Configuration ---
    with st.sidebar:
        st.markdown("""
        ## **Instructions:**
        #### 1. Enter your model's `/invocations` endpoint URL.
        #### 2. Ask a question in the text box.
        #### 3. Click **Send** button to receive an AI-generated answer.

        ## **Example URL:** 
        #### `https://localhost:5000/invocations`
        """)
        
        st.markdown("## ⚙️ Configuration")
        
        api_url = st.text_input(
            "MLflow Endpoint URL",
            value="https://localhost:57259/invocations",
            help="The full URL to the MLflow model's `/invocations` endpoint.",
        )
        
        force_regenerate = st.checkbox(
            "Force Regeneration",
            value=False,
            help="Bypass cache and force a new answer from the model."
        )

        st.markdown("---")
        st.info("This interface allows you to interact with the Multimodal RAG model.")

    # --- Main Chat Interface ---

    # Create a row for the logos in the top-left corner
    logo_col1, logo_col2, _ = st.columns([2, 2, 10]) 
    with logo_col1:
        st.image("assets/hp_logo.png", width=60)
    with logo_col2:
        st.image("assets/ai_studio_helix.png", width=60)

    st.markdown("<h1 style='text-align: center;'>🤖 ADO Wiki AI Assistant</h1>", unsafe_allow_html=True)
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display prior chat messages
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            # Display text content
            st.markdown(message["content"])
            
            if "metrics" in message and message["metrics"]:
                metrics = message["metrics"]
                gen_time = metrics.get("gen_time")
                faithfulness = metrics.get("faithfulness")
                relevance = metrics.get("relevance")

                if gen_time is not None and faithfulness is not None and relevance is not None:
                    st.markdown("---")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    metric_col1.metric(label="Generation Time", value=f"{gen_time:.2f} s")
                    metric_col2.metric(label="Faithfulness", value=f"{faithfulness:.2f}")
                    metric_col3.metric(label="Relevance", value=f"{relevance:.2f}")
            
            # Display images if they exist for an assistant message
            if "images" in message and message["images"]:
                st.markdown("<h4 class='image-gallery-header'>Retrieved Images</h4>", unsafe_allow_html=True)
                cols = st.columns(4)
                for idx, img_path in enumerate(message["images"]):
                    with cols[idx % 4]:
                        st.image(str(img_path), use_column_width=True)

    # Handle new user input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to session state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # Process and display the assistant's response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🧠 Thinking... Contacting the model..."):
                response = call_mlflow_api(api_url, prompt, force_regenerate)
                
                response_placeholder = st.empty()
                full_response_content = ""
                retrieved_images = []
                
                assistant_metrics = {}

                if response.get("success"):
                    data = response.get("data", {})
                    reply_text = data.get('reply', 'Sorry, I could not generate a reply.')
                    
                    full_response_content = reply_text
                    response_placeholder.markdown(full_response_content)
                    
                    gen_time = data.get("generation_time_seconds")
                    faithfulness = data.get("faithfulness")
                    relevance = data.get("relevance")

                    assistant_metrics = {
                        "gen_time": gen_time,
                        "faithfulness": faithfulness,
                        "relevance": relevance,
                    }
                    
                    used_images_str = data.get("used_images")
                    if used_images_str and isinstance(used_images_str, str):
                        try:
                            # Safely parse the string into a list of paths
                            image_path_list = ast.literal_eval(used_images_str)

                            if image_path_list and isinstance(image_path_list, list):
                                st.markdown("<h4 class='image-gallery-header'>Retrieved Images</h4>", unsafe_allow_html=True)
                                cols = st.columns(4)
                                for idx, rel_path_str in enumerate(image_path_list):
                                    img_filename = Path(rel_path_str).name
                                    full_img_path = IMAGE_DIR / img_filename
                                    if full_img_path.is_file():
                                        retrieved_images.append(full_img_path)
                                        with cols[idx % 4]:
                                            st.image(str(full_img_path), caption=img_filename, use_column_width=True)
                                    else:
                                        st.warning(f"Image not found: {img_filename}")

                        except (ValueError, SyntaxError):
                            st.warning("Could not parse image list from API response.")
                                
                    st.markdown("<h4 class='image-gallery-header'>Performance & Evaluation Metrics</h4>", unsafe_allow_html=True)
                    if gen_time is not None and faithfulness is not None and relevance is not None:
                        st.markdown("---")
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        metric_col1.metric(label="Generation Time", value=f"{gen_time:.2f} s")
                        metric_col2.metric(label="Faithfulness", value=f"{faithfulness:.2f}")
                        metric_col3.metric(label="Relevance", value=f"{relevance:.2f}")
                    

                else:
                    error_message = response.get("error", "An unknown error occurred.")
                    full_response_content = f"**Error:**\n\n{error_message}"
                    response_placeholder.error(full_response_content)

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response_content,
                    "images": retrieved_images,
                    "metrics": assistant_metrics, # Store metrics
                })

if __name__ == "__main__":
    main()