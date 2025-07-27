# app.py
import streamlit as st
import requests
import json
import pandas as pd
import warnings
from pathlib import Path

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
    """Applies custom CSS for HP branding and a rounded, modern chat interface."""
    css = """
    <style>
        /* Import a clean, modern font */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

        html, body, [class*="st-"] {
            font-family: 'Roboto', sans-serif;
        }

        /* Main app background */
        .stApp {
            background-color: #F0F2F5;
        }

        /* --- CHAT STYLES (UPDATED FOR ROUNDED LOOK) --- */
        
        /* Main container for chat messages */
        .st-emotion-cache-1f1G2gn {
            width: 100%;
        }

        /* Chat message bubbles */
        [data-testid="stChatMessage"] {
            padding: 1rem 1.25rem;
            border-radius: 22px; /* Increased rounding */
            margin-bottom: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.06);
            border: none;
            max-width: 85%; /* Bubbles don't span full width */
        }

        /* Assistant (AI) message styling - aligned left */
        div[data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"]) {
            background-color: #FFFFFF; /* White background for assistant */
            color: #262626;
            margin-right: auto;
        }
        
        /* User message styling - aligned right */
        div[data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"] p) {
            background-color: #0082C9; /* A slightly softer HP Blue for user messages */
            color: #FFFFFF;
            margin-left: auto;
        }
        
        /* Ensure the avatar and message content are properly aligned */
        [data-testid="stChatMessage"] > div {
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
        }

        /* Style for retrieved images container */
        .image-gallery-header {
            color: #333;
            font-weight: 500;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            padding-top: 1rem;
            border-top: 1px solid #E0E0E0;
        }
        
        /* --- SIDEBAR STYLES --- */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            padding: 1rem;
        }

        [data-testid="stSidebar"] .stMarkdown h2 {
            color: #0096D6; /* HP Blue for sidebar title */
            font-weight: 700;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #0096D6;
        }
        
        /* --- CHAT INPUT STYLES (UPDATED FOR ROUNDED LOOK) --- */
        [data-testid="stChatInput"] {
            background-color: #F0F2F5;
            border-top: 1px solid #D1D5DB;
        }
        
        [data-testid="stChatInput"] textarea {
            border-radius: 25px !important; /* Pill shape */
            border: 1px solid #D1D5DB !important;
            background-color: #FFFFFF !important;
        }

        /* Style send button */
        [data-testid="stChatInput"] button {
            background-color: #0096D6 !important;
            border-radius: 50% !important; /* Make it a circle */
            color: white !important;
        }
        
        [data-testid="stChatInput"] button:hover {
            background-color: #0076a8 !important;
        }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- API Interaction Logic (Unchanged) ---

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

# --- Main Application UI (Unchanged) ---

def main():
    """Renders the main HP-branded Chatbot application page."""
    load_css()
    
    # --- Sidebar for Configuration ---
    with st.sidebar:
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
    st.title("🤖 ADO Wiki AI Assistant")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display prior chat messages
    for message in st.session_state.messages:
        # Use a different avatar for user vs assistant
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            # Display text content
            st.markdown(message["content"])
            
            # Display images if they exist for an assistant message
            if "images" in message and message["images"]:
                st.markdown("<h4 class='image-gallery-header'>Retrieved Images</h4>", unsafe_allow_html=True)
                cols = st.columns(4)
                for idx, img_path in enumerate(message["images"]):
                    with cols[idx % 4]:
                        st.image(str(img_path), use_column_width=True)

    # Handle new user input
    if prompt := st.chat_input("Ask a question about your documents..."):
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

                if response.get("success"):
                    data = response.get("data", {})
                    reply_text = data.get('reply', 'Sorry, I could not generate a reply.')
                    
                    full_response_content = reply_text
                    response_placeholder.markdown(full_response_content)
                    
                    used_images_paths = data.get("used_images", [])
                    if used_images_paths:
                        st.markdown("<h4 class='image-gallery-header'>Retrieved Images</h4>", unsafe_allow_html=True)
                        cols = st.columns(4)
                        for idx, rel_path_str in enumerate(used_images_paths):
                            img_filename = Path(rel_path_str).name
                            full_img_path = IMAGE_DIR / img_filename
                            if full_img_path.is_file():
                                retrieved_images.append(full_img_path)
                                with cols[idx % 4]:
                                    st.image(str(full_img_path), caption=img_filename, use_column_width=True)
                            else:
                                st.warning(f"Image not found: {img_filename}")

                else:
                    error_message = response.get("error", "An unknown error occurred.")
                    full_response_content = f"**Error:**\n\n{error_message}"
                    response_placeholder.error(full_response_content)

                # Add the complete assistant message to session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response_content,
                    "images": retrieved_images
                })

if __name__ == "__main__":
    main()