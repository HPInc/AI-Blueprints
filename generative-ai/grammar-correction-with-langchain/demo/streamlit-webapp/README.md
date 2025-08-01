# How to Successfully Use the Streamlit Web App

## 1. Install Required Versions

Ensure that the following are installed on your machine:

- **Python** version **≥ 3.11** (https://www.python.org/downloads/)

## 2. Install Required Python Packages

Install the necessary Python packages by running:

```bash
pip install streamlit
```

## 3. Run the Streamlit App

To start the Streamlit app, execute the following command in your terminal:

```bash
streamlit run main.py
```

## 4. Select the Correct API Endpoint When Using the App

When interacting with the app:

- **Choose the exact and correct API URL** to connect to your deployed model.
- **Important:** The MLflow endpoint **must** use **HTTPS** (not HTTP).
- **Note:** In **Z by HP AI Studio**, the **port number** for your MLflow API **changes with each deployment**, so always verify the correct URL and port before starting a session.
