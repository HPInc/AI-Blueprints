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
IMAGE_DIR = Path("../../data/context/images")



# --- Page Configuration & Custom CSS ---

st.set_page_config(
    page_title="🤖 Multimodal RAG Chatbot",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def load_css():
    """Applies custom CSS for a branded, modern look."""
    css = """
    <style>
        /* Main app styling */
        .stApp {
            background-color: #FFFFFF;
        }

        /* Custom Header */
        .header {
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: black;
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3);
        }
        .header h1 {
            font-size: 2.5rem;
            font-weight: 600;
            margin: 0;
        }

        /* Styled containers for results */
        .result-container {
            background-color: #000000;
            border-left: 6px solid #007bff;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        }
        .result-container h2 {
            margin-top: 0;
            color: #0056b3;
            font-size: 1.5rem;
        }
        
        /* Custom Button */
        div[data-testid="stButton"] > button {
            background: linear-gradient(135deg, #28a745 0%, #218838 100%);
            color: black;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }
        div[data-testid="stButton"] > button:hover {
            box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4);
            transform: translateY(-2px);
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- API Interaction Logic ---

def call_mlflow_api(api_url: str, query: str, force_regenerate: bool) -> dict:
    """
    Calls the MLflow model serving endpoint.

    Args:
        api_url: The full URL to the /invocations endpoint.
        query: The user's question for the model.
        force_regenerate: Flag to control caching.

    Returns:
        A dictionary containing the prediction result or an error.
    """
    # MLflow expects a JSON object with a 'dataframe_records' key
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
        # Use verify=False for local deployments with self-signed certs
        response = requests.post(api_url, json=payload, headers=headers, verify=False, timeout=120)
        
        if response.status_code == 200:
            # The actual prediction is nested
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
    """Renders the main Streamlit application page."""
    load_css()
    
    # Render custom header
    st.markdown('<div class="header"><h1>🤖 AI Studio Multimodal Chatbot</h1></div>', unsafe_allow_html=True)

    # Initialize session state for persistence
    if "api_url" not in st.session_state:
        st.session_state.api_url = "http://127.0.0.1:5001/invocations" # Default local URL
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    # --- 1. Endpoint Configuration ---
    st.subheader("1. API Configuration")
    api_url = st.text_input(
        "MLflow Endpoint URL",
        value=st.session_state.api_url,
        help="The full URL to the MLflow model's `/invocations` endpoint.",
    )
    st.session_state.api_url = api_url # Persist changes

    st.markdown("---")

    # --- 2. Model Inputs ---
    st.subheader("2. Ask a Question")
    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_area(
            "Your Question:",
            height=150,
            placeholder="e.g., How do I manually clean my environment without hooh?",
            help="Enter the question you want to ask the RAG model."
        )

    with col2:
        force_regenerate = st.checkbox(
            "Force Regeneration",
            value=False,
            help="Check this box to bypass the semantic cache and force a new answer."
        )

    # --- 3. Submit Action ---
    if st.button("Get Answer"):
        # Validate inputs
        if not api_url.startswith(("http://", "https://")):
            st.error("Invalid Endpoint URL. Please enter a valid URL starting with `http://` or `https://`.")
        elif not query.strip():
            st.error("The question field cannot be empty. Please enter a question.")
        else:
            with st.spinner("🧠 Thinking... Contacting the model API, please wait."):
                result = call_mlflow_api(api_url, query, force_regenerate)
                st.session_state.last_result = result # Store result in session state

    st.markdown("---")

    # --- 4. Display Results ---
    st.subheader("3. Results")

    # Only try to display results if the 'last_result' object exists in the session
    if st.session_state.last_result:
        result = st.session_state.last_result

        # First, check if the API call was successful
        if result.get("success"):
            # If successful, we can safely access the 'data' key
            data = result.get("data", {})

            # --- Display Reply ---
            # --- Display Reply ---
            st.markdown(
                f"""
                <div class="result-container">
                    <h2>🤖 Model's Reply</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Get the reply and render it using st.markdown to process the ## and ** tags
            reply_text = data.get('reply', 'No reply text found in the response.')
            st.markdown(reply_text, unsafe_allow_html=True)

            # --- Display Used Images ---
            # The API returns a direct list, so we don't need json.loads()
            used_images = data.get("used_images", [])

            if used_images and isinstance(used_images, list):
                st.markdown(
                    """
                    <div class="result-container">
                        <h2>🖼️ Retrieved Images</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # Display images in columns for a cleaner layout
                cols = st.columns(4)
                col_index = 0

                for relative_path_str in used_images:
                    # The model returns a relative path; we need the filename.
                    image_filename = Path(relative_path_str).name
                    
                    # Construct the full, absolute path to the image
                    full_image_path = IMAGE_DIR / image_filename
                    # Check if the image file actually exists before trying to display it
                    if full_image_path.is_file():
                        with cols[col_index % 4]:
                            st.image(
                                str(full_image_path),
                                caption=image_filename,
                                use_column_width=True
                            )
                            col_index += 1
                    else:
                        # If the file is not found, print a warning
                        st.warning(f"Image not found: {image_filename}")
            else:
                st.info("No images were retrieved for this query.")
        else:
            # If the API call failed, display the error message
            error_message = result.get("error", "An unknown error occurred.")
            st.error(f"**Failed to get a response:**\n\n{error_message}")
if __name__ == "__main__":
    main()