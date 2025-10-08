import streamlit as st
import os
import requests
from pathlib import Path
import base64
import pandas as pd
import glob

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Movie Recommendation Agent", page_icon="🎬", layout="centered"
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem !important;
            max-width: 900px !important;
        }


        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            background: white;
        }


        .stApp {
            background: white;
        }


        /* Main Title Styling */
        .main-title {
            text-align: center;
            color: black;
            font-weight: 700;
            font-size: 3rem;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            animation: fadeIn 1s ease-in;
        }


        .subtitle {
            text-align: center;
            color: #000000;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            font-weight: 300;
        }


        /* Card Styling */
        .rating-card {
            background: linear-gradient(145deg, #ffffff, #f8f9fa);
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            margin: 20px 0;
            border: 1px solid rgba(255,255,255,0.8);
        }


        /* Button Styling */
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            font-size: 18px !important;
            font-weight: 600 !important;
            border-radius: 12px !important;
            padding: 12px 32px !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
        }


        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
        }


        /* Input Styling */
        .stNumberInput>div>div>input {
            font-size: 16px !important;
            padding: 12px !important;
            border-radius: 10px !important;
            border: 2px solid #e0e0e0 !important;
            transition: all 0.3s ease !important;
        }


        .stNumberInput>div>div>input:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
        }


        /* Rating Display Cards */
        .rating-item {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 15px 20px;
            border-radius: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            display: flex;
            align-items: center;
            transition: all 0.3s ease;
        }


        .rating-item:hover {
            transform: translateX(5px);
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15);
        }


        /* Recommendation Cards */
        .recommendation-card {
            background: linear-gradient(145deg, #ffffff, #f8f9fa);
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            margin: 15px 0;
            border-left: 6px solid #667eea;
            transition: all 0.3s ease;
        }


        .recommendation-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.2);
            border-left-color: #764ba2;
        }


        .recommendation-card h4 {
            color: #2C3E50;
            margin-bottom: 10px;
            font-weight: 600;
        }


        .score-badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 1.1rem;
        }


        /* Logo Container */
        .logo-container {
            display: flex;
            justify-content: space-around;
            align-items: center;
            margin-bottom: 2rem;
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }


        img[alt="HP Logo"],
        img[alt="AI Studio Logo"],
        img[alt="Z by HP Logo"] {
            width: 80px !important;
            height: auto !important;
            transition: all 0.3s ease;
        }


        img[alt="HP Logo"]:hover,
        img[alt="AI Studio Logo"]:hover,
        img[alt="Z by HP Logo"]:hover {
            transform: scale(1.1);
        }


        /* Section Headers */
        .section-header {
            color:  #000000;
            font-weight: 600;
            font-size: 1.3rem;
            margin: 20px 0 15px 0;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }


        /* Footer */
        .footer {
            text-align: center;
            color: #ffffff;
            margin-top: 3rem;
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }


        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }


        /* Success/Warning Messages */
        .stAlert {
            border-radius: 15px !important;
            border: none !important;
        }


        hr {
            border-color: rgba(255,255,255,0.3) !important;
            margin: 2rem 0 !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)



# --- Logo Section ---
def uri_from(path: Path) -> str:
    return (
        f"data:image/{path.suffix[1:].lower()};base64,"
        + base64.b64encode(path.read_bytes()).decode()
    )



assets = Path("assets")
hp_uri = uri_from(assets / "HP-Logo.png")
ais_uri = uri_from(assets / "AI-Studio.png")
zhp_uri = uri_from(assets / "Z-HP-logo.png")

st.markdown(
    f"""
    <div class="logo-container">
        <img src="{hp_uri}" alt="HP Logo">
        <img src="{ais_uri}" alt="AI Studio Logo">
        <img src="{zhp_uri}" alt="Z by HP Logo">
    </div>
""",
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown(
    """
    <h1 class="main-title">🎬 Movie Recommendation Agent</h1>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# MLflow API Configuration
# ─────────────────────────────────────────────────────────────
MLFLOW_ENDPOINT = "http://localhost:5002/invocations"
api_url = MLFLOW_ENDPOINT


# Load movie titles with improved MLflow artifact handling
@st.cache_data
def load_movie_titles_from_mlflow():
    """
    Load movie titles from MLflow model artifacts using multiple fallback strategies.
    This mimics the load_context behavior from your MLflow model.
    """
    import mlflow
    from mlflow import MlflowClient
    
    try:
        # Method 1
        try:
            mlflow.set_tracking_uri("/phoenix/mlflow")
            client = MlflowClient()
            model_name = "movie_titles"
            
            # Get the latest version
            model_metadata = client.get_latest_versions(model_name, stages=["None"])
            if model_metadata:
                latest_version = model_metadata[0].version
                model_uri = f"models:/{model_name}/{latest_version}"
                
                # Download the model artifacts
                local_path = mlflow.artifacts.download_artifacts(model_uri)
                movie_titles_path = os.path.join(local_path, "movie_titles.csv")
                
                if os.path.exists(movie_titles_path):
                    df = pd.read_csv(movie_titles_path)
                    if not df.empty and 'item_id' in df.columns and 'title' in df.columns:
                        st.success("✅ Movie titles loaded from MLflow model registry")
                        return df
        except Exception as e:
            st.warning(f"Could not load from MLflow model registry: {str(e)}")
        
        # Method 2
        mlflow_paths = [
            "/phoenix/mlflow/*/*/artifacts/movie_titles/artifacts/movie_titles.csv",
            "/phoenix/mlflow/*/*/artifacts/movie_titles/movie_titles.csv"
        ]
        
        
        for pattern in mlflow_paths:
            matching_paths = glob.glob(pattern)
            for mlflow_path in matching_paths:
                if os.path.exists(mlflow_path):
                    df = pd.read_csv(mlflow_path)
                    if not df.empty and 'item_id' in df.columns and 'title' in df.columns:
                        st.info(f"📁 Movie titles loaded from MLflow artifact: {os.path.basename(mlflow_path)}")
                        return df
        
        st.warning("⚠️ Could not find movie titles file in any expected location")
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading movie titles: {str(e)}")
        return None

# Load movie titles using the method
movie_titles_df = load_movie_titles_from_mlflow()

def get_movie_title(movie_id):
    """
    Get movie title for a given movie ID with fallback handling.
    """
    global movie_titles_df
    
    # Try primary method first  
    if movie_titles_df is not None:
        title_row = movie_titles_df[movie_titles_df['item_id'] == movie_id]
        if not title_row.empty:
            return title_row.iloc[0]['title'] 
    return f"Movie ID {movie_id}"

# ─────────────────────────────────────────────────────────────
# Main – Data Input
# ─────────────────────────────────────────────────────────────

st.markdown(
    "<p style='text-align: center; color: #666;'>Add 1-10 movies with your ratings to get personalized recommendations</p>",
    unsafe_allow_html=True,
)

# Initialize session state for movie ratings
if "movie_ratings" not in st.session_state:
if "movie_ratings" not in st.session_state:
    st.session_state.movie_ratings = []

# Form to add movie ratings
with st.form("add_rating_form"):
    col1, col2 = st.columns(2)
    with col1:
        movie_id = st.number_input("🎬 Movie ID", min_value=1, value=1)
        # Show movie title if available
        movie_title = get_movie_title(movie_id)
    with col2:
        rating = st.number_input(
            "⭐ Your Rating", min_value=0.5, max_value=5.0, step=0.5, value=3.0
        )

        rating = st.number_input(
            "⭐ Your Rating", min_value=0.5, max_value=5.0, step=0.5, value=3.0
        )

    submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
    with submit_col2:
        submitted = st.form_submit_button("➕ Add Rating", use_container_width=True)


    if submitted:
        if len(st.session_state.movie_ratings) < 10:
            existing_ids = [r["movie_id"] for r in st.session_state.movie_ratings]
            if movie_id not in existing_ids:
                movie_title = get_movie_title(movie_id)
                title_display = movie_title if movie_title else f"Movie ID {movie_id}"
                st.session_state.movie_ratings.append(
                    {"movie_id": movie_id, "rating": rating, "title": title_display}
                )
                st.success(f"✅ Added {title_display} with {rating} ⭐ rating!")
            else:
                st.warning("⚠️ You've already rated this movie!")
        else:
            st.warning("⚠️ Maximum 10 movies allowed!")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Display current ratings
if st.session_state.movie_ratings:
    st.markdown(
        '<p class="section-header">📝 Your Current Ratings</p>', unsafe_allow_html=True
    )
   
    for i, rating_data in enumerate(st.session_state.movie_ratings):
        movie_id = rating_data["movie_id"]
        rating = rating_data["rating"]
        title = get_movie_title(movie_id) or "Unknown Title"

        col1, col2, col3 = st.columns([4, 3, 1])
        with col1:
            st.markdown(f"**🎬 {title}** (ID: {movie_id})")
        with col2:
            st.markdown(f"**⭐ Rating:** {rating}")
        with col3:
            if st.button("🗑️", key=f"delete_{i}", help="Remove this rating"):
                st.session_state.movie_ratings.pop(i)
                st.rerun()


        if i < len(st.session_state.movie_ratings) - 1:
            st.markdown(
                "<hr style='margin: 10px 0; border-color: #e0e0e0;'>",
                unsafe_allow_html=True,
            )


    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Get Recommendations Button
# ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🍿 Get My Recommendations", type="primary"):
    if not st.session_state.movie_ratings:
        st.warning("⚠️ Please add at least one movie rating to get recommendations!")
    else:
        with st.spinner("🎯 Analyzing your taste and finding perfect matches..."):
            movie_ids = [
                rating_data["movie_id"]
                for rating_data in st.session_state.movie_ratings
            ]
            ratings = [
                rating_data["rating"] for rating_data in st.session_state.movie_ratings
            ]

            payload = {"inputs": {"movie_id": movie_ids, "rating": ratings}}

            movie_ids = [
                rating_data["movie_id"]
                for rating_data in st.session_state.movie_ratings
            ]
            ratings = [
                rating_data["rating"] for rating_data in st.session_state.movie_ratings
            ]

            payload = {"inputs": {"movie_id": movie_ids, "rating": ratings}}

            try:
                response = requests.post(api_url, json=payload, verify=False)
                response.raise_for_status()
                data = response.json()

                if "predictions" in data:
                    st.markdown(
                        '<p class="section-header">🎭 Your Personalized Recommendations</p>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        '<p class="section-header">🎭 Your Personalized Recommendations</p>',
                        unsafe_allow_html=True,
                    )

                    for i, movie in enumerate(data["predictions"], 1):
                        title = movie[0]
                        score = movie[1]
                        st.markdown(
                            f"""
                            <div class="recommendation-card">
                                <h4>#{i}. 🍿 {title}</h4>
                                <p><strong>Match Score:</strong> <span class="score-badge">{score:.2f}</span></p>
                            </div>
                        """,
                            unsafe_allow_html=True,
                        )


                    st.markdown("<br>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button(
                            "🔄 Clear All & Start Over", use_container_width=True
                        ):
                        if st.button(
                            "🔄 Clear All & Start Over", use_container_width=True
                        ):
                            st.session_state.movie_ratings = []
                            st.rerun()


                else:
                    st.error("❌ Unexpected response format. Please try again.")

            except requests.exceptions.RequestException as e:
                st.error("❌ Error connecting to recommendation service.")
                st.error(str(e))
# ─────────────────────────────────────────────────────────────
# 4 ▸ Footer
# ─────────────────────────────────────────────────────────────
st.write("---")
st.write("Built with ❤️ using HP AI Studio")
