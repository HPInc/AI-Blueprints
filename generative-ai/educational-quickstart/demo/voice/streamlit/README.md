# Voice Assistant Demo — Streamlit App

This folder contains the focused Streamlit front-end for the **AIStudio-EQ-Voice** registered model.

## Start the model server first

```bash
mlflow models serve -m models:/AIStudio-EQ-Voice/1 -p 5002 --no-conda
```

## Launch the app

```bash
cd demo/voice
python -m poetry install
python -m poetry run streamlit run main.py
```

## What this app does

- Accepts audio file upload (WAV, MP3, OGG, FLAC) or text command fallback
- Encodes audio as base64 and sends with `question` field to the VoiceModel
- Displays Whisper transcription + LLM response
