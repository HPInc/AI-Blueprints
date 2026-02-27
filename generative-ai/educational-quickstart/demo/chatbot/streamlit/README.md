# Chatbot Demo — Streamlit App

This folder contains the focused Streamlit front-end for the **AIStudio-EQ-Chatbot** registered model.

## Start the model server first

```bash
mlflow models serve -m models:/AIStudio-EQ-Chatbot/1 -p 5002 --no-conda
```

## Launch the app

```bash
cd demo/chatbot
python -m poetry install
python -m poetry run streamlit run main.py
```

## What this app does

- Sends `question` + `system_prompt` to the ChatbotModel via HTTP POST
- Displays the LLM response and conversation history
- System prompt is editable in the sidebar — change the LLM's personality
