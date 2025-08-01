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

To start the Streamlit app, cd to the directory containing `main.py` or `main_no_dbcache.py` and execute the following command in your terminal:

`main.py` is the recommended version as it uses the database cache for faster performance. If you want to run the app without database caching, use `main_no_dbcache.py`. The `main_no_dbcache.py` does not use the database cache, which may result in slower performance since the data is not saved to be queried.

```bash
streamlit run main.py
```

```bash
streamlit run main_no_dbcache.py
```

## 4. Select the Correct API Endpoint When Using the App

When interacting with the app:

- **Choose the exact and correct API URL** to connect to your deployed model.
- **Important:** The MLflow endpoint **must** use **HTTPS** (not HTTP).
- **Note:** In **Z by HP AI Studio**, the **port number** for your MLflow API **changes with each deployment**, so always verify the correct URL and port before starting a session.
