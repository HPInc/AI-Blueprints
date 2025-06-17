# How to Successfully Use the Streamlit Web App

## 1. Install Required Versions
Ensure that the following are installed on your machine:
- **Python** version **≥ 3.11** (https://www.python.org/downloads/)
- **Poetry** version **≥ 2.0.0 and < 3.0.0** (https://python-poetry.org/docs/)

## 2. Set Up the Virtual Environment and Install Dependencies
Navigate to the project's root directory and run the following command to set up a virtual environment using Poetry and install all required packages:
```bash
python -m poetry install
```

## 3. Launch the Streamlit Web App
Still in the project's root directory, start the Streamlit app by running:
```bash
python -m poetry run streamlit run "main.py"
```

## 4. Select the Correct API Endpoint When Using the App
When interacting with the app:
- **Choose the exact and correct API URL** to connect to your deployed model.
- **Important:** The MLflow endpoint **must** use **HTTPS** (not HTTP).
- **Note:** In **Z by HP AI Studio**, the **port number** for your MLflow API **changes with each deployment**, so always verify the correct URL and port before starting a session.

## 5. Choose a MNIST Dataset Image to Upload on Streamlit
- The images can be found at https://huggingface.co/datasets/ylecun/mnist

---

> Built with ❤️ using [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).