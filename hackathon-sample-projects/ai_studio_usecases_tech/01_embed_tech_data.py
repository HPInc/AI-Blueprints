import os
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Define constants
DATA_FILE = "data/tech_adoption_dataset.csv"
EMBEDDINGS_FILE = "data/tech_embeddings.npy"

def load_or_download_model():
    """Load or download the sentence transformer model"""
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_DIR = "model/sentence-transformer"
    
    # Create model directory if it doesn't exist
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(exist_ok=True, parents=True)
    
    print("Loading sentence transformer model...")
    try:
        # Try to load model from the local directory
        model = SentenceTransformer(MODEL_DIR)
        print("Sentence transformer model loaded from local directory!")
    except Exception as e:
        print(f"Error loading model from local path: {str(e)}")
        print(f"Downloading model {MODEL_NAME} from Hugging Face...")
        model = SentenceTransformer(MODEL_NAME)
        
        # Save the model to our specific directory
        model.save(MODEL_DIR)
        print(f"Model downloaded and saved to: {MODEL_DIR}")
    
    return model

def generate_embeddings(model, texts, batch_size=32):
    """Generate embeddings for the texts"""
    print(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings

def main():
    """Generate embeddings for technology user descriptions"""
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Check for data file
    if not os.path.exists(DATA_FILE):
        print(f"Tech adoption dataset not found at {DATA_FILE}")
        print("Please run the data generation script (00_generate_tech_adoption_data.py) first.")
        return
    
    # Load the model
    model = load_or_download_model()
    
    # Load data
    print(f"Loading technology adoption data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded data for {len(df)} users")
    
    # Check if we have pre-formatted descriptions
    if "user_description" not in df.columns:
        print("Error: Dataset doesn't contain pre-formatted user descriptions.")
        print("Please re-run the data generation script to create user descriptions.")
        return
    
    # Get descriptions and generate embeddings
    descriptions = df["user_description"].tolist()
    embeddings = generate_embeddings(model, descriptions)
    
    # Save embeddings
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"Saved embeddings to {EMBEDDINGS_FILE}")
    print("Embedding process completed successfully!")

if __name__ == "__main__":
    main() 