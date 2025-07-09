import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple

@st.cache_resource
def load_artifacts() -> Tuple[object, object]:
    """Load the trained model and scaler once and cache them."""
    artifacts_dir = Path(__file__).resolve().parent.parent.parent / "artifacts"
    model_path = artifacts_dir / "model.pkl"
    scaler_path = artifacts_dir / "scaler.pkl"

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def predict_species(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
) -> str:
    """Run inference using the loaded model and scaler."""
    model, scaler = load_artifacts()
    input_df = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=["sepal-length", "sepal-width", "petal-length", "petal-width"],
    )
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    return str(prediction)


def main() -> None:
    """Streamlit front-end for interactive inference."""
    st.set_page_config(
        page_title="Iris Flower Classifier",
        page_icon="🌸",
        layout="centered",
    )

    # ---- Global CSS (deeper gray bg + widened card + fonts + logo + buttons) ---- #
    st.markdown(
        """
        <style>
          /* Base font size for everything except h1 */
          .stApp, .stApp *:not(h1) {
            font-size: 18px !important;
            line-height: 1.4 !important;
          }

          /* App background */
          .stApp {
            background-color: #cccccc;
            font-family: 'Segoe UI', sans-serif;
          }

          /* Widened card (~75% viewport) */
          .block-container {
            background-color: rgba(255,255,255,0.94) !important;
            width: 75vw !important;
            max-width: none !important;
            padding: 3rem 2rem !important;
            border-radius: 18px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            margin: 2rem auto;
          }

          /* Logo styling */
          .logo-container {
            text-align: center;
            margin: 1.5rem 0;
          }

          /* Title – keep this big */
          h1 {
            color: #334f8d;
            text-align: center;
            font-weight: 800;
            font-size: 2.8rem;
            margin-bottom: 0.5rem;
          }

          /* Button styling */
          .stButton > button {
            background-color: #334f8d;
            color: white;
            border-radius: 8px;
            padding: 0.8rem 1.6rem;
            font-size: 1.1rem;
            transition: 0.3s;
          }
          .stButton > button:hover {
            background-color: #5063a6;
          }

          /* Success box */
          .stAlert > div {
            background-color: #e8f0fe;
            border-left: 6px solid #334f8d;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- Logo ---- #
    logo_path = Path(__file__).resolve().parent.parent / "assets" / "Z-HP-D93Sc3af.png"
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image(str(logo_path), width=180, use_column_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Core UI ---- #
    st.title("🌸 Iris Flower Classifier")
    st.write("Provide flower measurements and get the predicted Iris species.")

    st.header("Input Features (cm)")
    sepal_length = st.slider("Sepal length", 4.0, 10.0, 5.4, 0.1)
    sepal_width = st.slider("Sepal width", 2.0, 10.0, 3.4, 0.1)
    petal_length = st.slider("Petal length", 1.0, 10.0, 1.3, 0.1)
    petal_width = st.slider("Petal width", 0.1, 10.0, 0.2, 0.1)

    if st.button("Predict Species"):
        species = predict_species(
            sepal_length, sepal_width, petal_length, petal_width
        )
        st.success(f"Predicted Iris Species: **{species}**")


if __name__ == "__main__":
    main()
