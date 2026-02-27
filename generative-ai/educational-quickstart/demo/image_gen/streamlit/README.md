# Image Generator Demo — Streamlit App

This folder contains the focused Streamlit front-end for the **AIStudio-EQ-ImageGen** registered model.

## Start the model server first

```bash
mlflow models serve -m models:/AIStudio-EQ-ImageGen/1 -p 5002 --no-conda
```

## Launch the app

```bash
cd demo/image_gen
python -m poetry install
python -m poetry run streamlit run main.py
```

## What this app does

- Sends a text `prompt` to the ImageGenModel via HTTP POST
- Receives a base64-encoded PNG image and displays it inline
- Uses SDXL-Turbo with 4 denoising steps for fast generation
