# Document Analyzer Demo — Streamlit App

This folder contains the focused Streamlit front-end for the **AIStudio-EQ-Document** registered model.

## Start the model server first

```bash
mlflow models serve -m models:/AIStudio-EQ-Document/1 -p 5002 --no-conda
```

## Launch the app

```bash
cd demo/document
python -m poetry install
python -m poetry run streamlit run main.py
```

## What this app does

- Accepts document upload (.txt, .csv, .md) or text paste
- Sends `question` + `input_text` to the DocumentModel via HTTP POST
- Displays the RAG-synthesized answer and analysis details
