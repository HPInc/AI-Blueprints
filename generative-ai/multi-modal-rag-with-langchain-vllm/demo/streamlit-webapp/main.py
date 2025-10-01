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


def call_model_api(api_url: str, command: str, data_payload: dict) -> dict:
    """Calls the MLflow model serving endpoint with a specific command."""
    record = {"command": command, **data_payload}
    payload = {"dataframe_records": [record]}
    headers = {"Content-Type": "application/json"}

    try:
        timeout = 300 if command == "update_kb" else 120
        response = requests.post(
            api_url, json=payload, headers=headers, verify=False, timeout=timeout
        )

        if response.status_code == 200:
            predictions = response.json().get("predictions", [])
            if predictions:
                return {"success": True, "data": predictions[0]}
            else:
                return {"success": False, "error": "API returned an empty prediction."}
        else:
            return {
                "success": False,
                "error": f"API Error (Status {response.status_code}): {response.text}",
            }

    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Network or connection error: {e}"}


# --- Main Application UI ---


def main():
    """Renders the main HP-branded Chatbot application page."""

    # Initialize session state
    if "kb_ready" not in st.session_state:
        st.session_state.kb_ready = False
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Sidebar for Configuration ---
    with st.sidebar:
        st.markdown("## ⚙️ Endpoint Configuration")
        api_url = st.text_input(
            "MLflow Endpoint URL",
            value="https://localhost:5000/invocations",
            help="The full URL to the MLflow model's `/invocations` endpoint.",
        )
        st.markdown("---")

        with st.form("kb_form"):
            st.markdown("## 📚 Knowledge Base Source")
            st.info("Enter your Azure DevOps details to sync the wiki.")

            ado_org = st.text_input("ADO Organization", placeholder="e.g., MyCompany")
            ado_project = st.text_input("ADO Project", placeholder="e.g., MyProject")
            ado_wiki = st.text_input(
                "ADO Wiki Name", placeholder="e.g., MyProject.wiki"
            )
            ado_pat = st.text_input("ADO PAT", type="password")

            submitted = st.form_submit_button("Sync & Load Knowledge Base")

            if submitted:
                if not all([ado_org, ado_project, ado_wiki, ado_pat]):
                    st.warning("Please fill in all ADO fields.")
                else:
                    with st.spinner(
                        "Connecting to ADO and building knowledge base... This may take several minutes."
                    ):
                        # Construct the payload for the 'update_kb' command
                        config_payload = {
                            "config": {
                                "AZURE_DEVOPS_ORG": ado_org,
                                "AZURE_DEVOPS_PROJECT": ado_project,
                                "AZURE_DEVOPS_WIKI_IDENTIFIER": ado_wiki,
                            },
                            "secrets": {"AIS_ADO_TOKEN": ado_pat},
                        }
                        update_payload = {"payload": json.dumps(config_payload)}
                        response = call_model_api(api_url, "update_kb", update_payload)

                        # --- FIX: Check both the outer 'success' and the inner 'status' key ---
                        if (
                            response.get("success")
                            and response.get("data", {}).get("status") == "success"
                        ):
                            st.session_state.kb_ready = True
                            st.success(
                                response["data"].get(
                                    "message", "Knowledge base loaded successfully!"
                                )
                            )
                        else:
                            st.session_state.kb_ready = False
                            # Extract the specific error message from the model's response
                            if response.get(
                                "success"
                            ):  # The API call worked, but the command failed
                                error_msg = response.get("data", {}).get(
                                    "message",
                                    "An unknown error occurred inside the model.",
                                )
                            else:  # The API call itself failed
                                error_msg = response.get(
                                    "error", "A network or unknown error occurred."
                                )

                            st.error(f"Failed to load knowledge base: {error_msg}")
        st.markdown("---")

        if st.session_state.kb_ready:
            st.markdown("## 💬 Chat Options")
            force_regenerate = st.checkbox(
                "Force Regeneration", value=False, help="Bypass cache for a new answer."
            )

    # --- Main Page Branding ---
    logo_col1, logo_col2, _ = st.columns([1, 1, 10])
    with logo_col1:
        st.image("assets/hp_logo.png", width=60)
    with logo_col2:
        st.image("assets/ai_studio_helix.png", width=60)

    st.markdown(
        "<h1 style='text-align: center;'>🤖 ADO Wiki AI Assistant</h1>",
        unsafe_allow_html=True,
    )

    # --- Main Chat Interface ---
    if st.session_state.kb_ready:
        # Display prior chat messages from history
        for message in st.session_state.messages:
            avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

                # Display metrics stored in history
                if "metrics" in message and message["metrics"]:
                    metrics = message["metrics"]
                    gen_time, faithfulness, relevance = (
                        metrics.get("gen_time"),
                        metrics.get("faithfulness"),
                        metrics.get("relevance"),
                    )
                    if all(v is not None for v in [gen_time, faithfulness, relevance]):
                        st.markdown("---")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Generation Time", f"{gen_time:.2f} s")
                        c2.metric("Faithfulness", f"{faithfulness:.2f}")
                        c3.metric("Relevance", f"{relevance:.2f}")

                # Display images stored in history
                if "images" in message and message["images"]:
                    st.markdown(
                        "<h4 class='image-gallery-header'>Retrieved Images</h4>",
                        unsafe_allow_html=True,
                    )
                    cols = st.columns(min(len(message["images"]), 4))
                    for idx, img_path in enumerate(message["images"]):
                        with cols[idx % 4]:
                            if Path(img_path).is_file():
                                st.image(str(img_path), use_container_width=True)
                            else:
                                st.warning(f"Image not found at {Path(img_path).name}")

        # Handle new user input
        if prompt := st.chat_input("Ask a question about the wiki..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🧠 Thinking..."):
                    query_payload = {
                        "query": prompt,
                        "force_regenerate": force_regenerate,
                    }
                    response = call_model_api(api_url, "query", query_payload)

                    full_response_content, retrieved_images, assistant_metrics = (
                        "",
                        [],
                        {},
                    )
                    if response.get("success"):
                        data = response.get("data", {})
                        full_response_content = data.get(
                            "reply", "Sorry, I could not generate a reply."
                        )
                        assistant_metrics = {
                            "gen_time": data.get("generation_time_seconds"),
                            "faithfulness": data.get("faithfulness"),
                            "relevance": data.get("relevance"),
                        }
                        st.markdown(full_response_content)

                        # Process and display images for the current response
                        used_images_json = data.get("used_images", "[]")
                        try:
                            retrieved_images_b64 = json.loads(used_images_json)
                            if retrieved_images_b64:
                                st.markdown(
                                    "<h4 class='image-gallery-header'>Retrieved Images</h4>",
                                    unsafe_allow_html=True,
                                )
                                cols = st.columns(min(len(retrieved_images_b64), 4))
                                for idx, b64_string in enumerate(retrieved_images_b64):
                                    with cols[idx % 4]:
                                        image_bytes = base64.b64decode(b64_string)
                                        st.image(image_bytes, use_container_width=True)
                        except (json.JSONDecodeError, TypeError):
                            st.warning("Could not parse image data from API.")

                        st.markdown("---")
                        c1, c2, c3 = st.columns(3)
                        c1.metric(
                            "Generation Time", f"{assistant_metrics['gen_time']:.2f} s"
                        )
                        c2.metric(
                            "Faithfulness", f"{assistant_metrics['faithfulness']:.2f}"
                        )
                        c3.metric("Relevance", f"{assistant_metrics['relevance']:.2f}")

                    else:
                        full_response_content = f"**Error:** {response.get('error')}"
                        st.error(full_response_content)

                    # Append the complete message to history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_response_content,
                            "images": retrieved_images,
                            "metrics": assistant_metrics,
                        }
                    )
    else:
        st.info(
            "👋 Welcome! Please sync a knowledge base using the sidebar to begin the chat."
        )


if __name__ == "__main__":
    main()
