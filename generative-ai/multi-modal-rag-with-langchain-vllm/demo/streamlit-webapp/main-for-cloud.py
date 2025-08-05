# app.py
import streamlit as st
import requests
import json
import pandas as pd
import warnings
from pathlib import Path
import ast
import base64
# Ignore SSL warnings for local development
warnings.filterwarnings("ignore", message="Unverified HTTPS request")


# --- Page Configuration & Custom CSS ---

st.set_page_config(
    page_title="ADO Wiki AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- API Interaction Logic ---
def call_model_api(api_url: str, query: str, config_payload: dict) -> dict:
    """Calls the MLflow model serving endpoint."""
    record = {
        "query": query,
        "payload": json.dumps(config_payload)
    }
    payload = {"dataframe_records": [record]}
    headers = {"Content-Type": "application/json"}
    
    try:
        # Increased timeout for the stateless operation
        response = requests.post(api_url, json=payload, headers=headers, verify=False, timeout=300)
        
        if response.status_code == 200:
            predictions = response.json().get("predictions", [])
            if predictions:
                return {"success": True, "data": predictions[0]}
            return {"success": False, "error": "API returned an empty prediction."}
        else:
            return {"success": False, "error": f"API Error (Status {response.status_code}): {response.text}"}
            
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Network or connection error: {e}"}


# --- Main Application UI ---
def main():
    """Renders the main Chatbot application page."""

    api_url = "https://7afcf880ce75.ngrok-free.app/invocations"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Sidebar for Configuration ---
    with st.sidebar:
        st.markdown("## 📚 Knowledge Base Source")
        st.info("Enter your Azure DevOps details. These are required for each query.")
        
        ado_org = st.text_input("ADO Organization", key="ado_org", placeholder="e.g., MyCompany")
        ado_project = st.text_input("ADO Project", key="ado_project", placeholder="e.g., MyProject")
        ado_wiki = st.text_input("ADO Wiki Name", key="ado_wiki", placeholder="e.g., MyProject.wiki")
        ado_pat = st.text_input("ADO PAT", key="ado_pat", type="password", help="Your Personal Access Token for ADO.")

    # --- Main Page Branding ---
    logo_col1, logo_col2, _ = st.columns([2, 2, 10]) 
    with logo_col1:
        st.image("assets/hp_logo.png", width=60)
    with logo_col2:
        st.image("assets/ai_studio_helix.png", width=60)
    st.markdown("<h1 style='text-align: center;'>🤖 ADO Wiki AI Assistant</h1>", unsafe_allow_html=True)
    
    # --- Main Chat Interface (Single source of truth for rendering) ---
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            
            # If the message is from the assistant, display its metrics and images
            if message["role"] == "assistant" and message.get("data"):
                data = message["data"]
                
                # Display Metrics
                st.markdown("---")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Pipeline Time", f"{data.get('total_pipeline_time_seconds', 0):.2f} s")
                c2.metric("Generation Time", f"{data.get('generation_time_seconds', 0):.2f} s")
                c3.metric("Faithfulness", f"{data.get('faithfulness', 0) * 100:.0f}%")
                c4.metric("Relevance", f"{data.get('relevance', 0) * 100:.0f}%")
                                
                # Display Images
                try:
                    images_b64 = json.loads(data.get("used_images", "[]"))
                    if images_b64:
                        st.markdown("<h4 class='image-gallery-header'>Retrieved Images</h4>", unsafe_allow_html=True)
                        cols = st.columns(min(len(images_b64), 4))
                        for idx, b64_string in enumerate(images_b64):
                            with cols[idx % 4]:
                                st.image(base64.b64decode(b64_string), use_container_width=True)
                except (json.JSONDecodeError, TypeError):
                    st.warning("Could not parse image data from API response.")

    # --- Handle new user input ---
    if prompt := st.chat_input("Ask a question about the wiki..."):
        # 1. Validate credentials
        if not all([api_url, ado_org, ado_project, ado_wiki, ado_pat]):
            st.error("Please fill in all configuration fields in the sidebar.")
            st.stop()

        # Add user message to state and immediately display it
        st.session_state.messages.append({"role": "user", "content": prompt, "data": None})
        
        # Construct payload and call API
        with st.spinner("🧠 Fetching database, thinking, and cleaning up... This may take a moment."):
            config_payload = {
                "config": {"AZURE_DEVOPS_ORG": ado_org, "AZURE_DEVOPS_PROJECT": ado_project, "AZURE_DEVOPS_WIKI_IDENTIFIER": ado_wiki},
                "secrets": {"AIS_ADO_TOKEN": ado_pat}
            }
            response = call_model_api(api_url, prompt, config_payload)
        
        # Prepare assistant's message content and data
        assistant_message = {
            "role": "assistant",
            "content": "",
            "data": None
        }
        if response.get("success"):
            assistant_message["data"] = response.get("data", {})
            assistant_message["content"] = assistant_message["data"].get('reply', 'Sorry, I could not generate a reply.')
        else:
            assistant_message["content"] = f"**Error:** {response.get('error')}"
            
        # Add assistant message to state
        st.session_state.messages.append(assistant_message)
        st.rerun()
if __name__ == "__main__":
    main()