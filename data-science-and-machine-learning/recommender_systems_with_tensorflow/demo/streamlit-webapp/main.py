import streamlit as st
import pandas as pd
# import mlflow.pyfunc


st.title("🎥 Movies Recommender")

@st.cache_resource
def load_model():
    model_path = ""
    # return mlflow.pyfunc.load_model(model_path)

model = load_model()

user_id= st.number_input("Enter your ID", min_value=1, step=1)
movie_id = st.number_input("Enter movie ID", min_value=0, step=1)

if st.button("🍿 Recommend movies"):
    input_data = pd.DataFrame({
        "user_id": [int(user_id)],
        "movie_id": [int(movie_id)]
    })
    
    try:
        prediction = model.predict(input_data)
        movie_title = prediction[0]["movie_title"]
        score = prediction[0]["prediction"]

        st.success("Recommended films:")
        st.write(f"Title:{movie_title}")
        st.write(f"Rating:{score}")
    except Exception as e:
        st.error(f"Error in making prediction: {e}")