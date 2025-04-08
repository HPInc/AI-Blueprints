import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer


def download_model():
    """Download the sentence-transformers model if it doesn't exist locally"""
    # Define constants for model download
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_DIR = "model/sentence-transformer"
    
    # Create model directory if it doesn't exist
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if model already exists (checking for config.json as indicator)
    if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        print(f"Sentence transformer model already exists at: {MODEL_DIR}")
    else:
        print(f"Sentence transformer model not found locally. Downloading {MODEL_NAME}...")
        
        # Download the model by initializing it (it will be cached)
        model = SentenceTransformer(MODEL_NAME)
        
        # Save the model to our specific directory
        model.save(MODEL_DIR)
        print(f"Model downloaded and saved to: {MODEL_DIR}")

    print(f"Using sentence transformer model at: {MODEL_DIR}")
    return MODEL_DIR


def load_model():
    """Load the sentence-transformers model"""
    print("Loading sentence transformer model...")
    try:
        # Load model from the local directory
        model_dir = "model/sentence-transformer"
        model = SentenceTransformer(model_dir)
        print("Sentence transformer model loaded successfully!")
    except Exception as e:
        print(f"Error loading model from local path: {str(e)}")
        print("Downloading model from Hugging Face...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("Sentence transformer model loaded successfully from Hugging Face!")
    
    return model


def generate_embeddings(model, texts, batch_size=32):
    """Generate embeddings for texts using sentence-transformers"""
    print(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}")
    
    # SentenceTransformer's encode method handles batching internally
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    return embeddings


def embed_banking_data(model):
    """Generate and save embeddings for banking customer data"""
    # Setup paths
    data_file = "data/banking_dataset.csv"
    os.makedirs("data", exist_ok=True)
    
    # Check for data file
    if not os.path.exists(data_file):
        print(f"Banking dataset not found at {data_file}")
        return
    
    # Load and process data
    df = pd.read_csv(data_file)
    print(f"Processing {len(df)} customers")
    
    # Create descriptions - remove slice to process all customers
    descriptions = []
    for _, row in df.iterrows():  # Removed the slice to process all customers
        desc = f"Customer ID: {row['customer_id']}, Age: {row['age']}, Income: ${row['income']}, " + \
               f"Credit Score: {row['credit_score']}, Segment: {row['segment']}, Risk: {row['risk_profile']}"
        descriptions.append(desc)
    
    # Generate and save embeddings
    print(f"Generating embeddings for {len(descriptions)} customers")
    embeddings = generate_embeddings(model, descriptions)
    np.save("data/customer_embeddings.npy", embeddings)
    
    # Demo similarity search
    find_similar_customers(embeddings, descriptions, customer_id=1)
    print(f"Embeddings saved to data/customer_embeddings.npy")


def find_similar_customers(embeddings, descriptions, customer_id=1, top_n=3):
    """Find customers similar to the target customer based on embeddings"""
    # Find the index of the target customer (assuming customer_ids start at 1)
    target_idx = customer_id - 1
    
    if target_idx < 0 or target_idx >= len(embeddings):
        print(f"Customer ID {customer_id} not found in embeddings")
        return
    
    # Get the target embedding
    target_embedding = embeddings[target_idx]
    
    # Calculate cosine similarity using numpy operations
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Reshape to 2D arrays as required by sklearn
    target_embedding_2d = target_embedding.reshape(1, -1)
    all_embeddings = embeddings.reshape(embeddings.shape[0], -1)
    
    # Compute similarity
    similarities = cosine_similarity(target_embedding_2d, all_embeddings)[0]
    
    # Create list of (index, similarity) tuples, excluding the target customer
    similarity_scores = [(i, similarities[i]) for i in range(len(similarities)) if i != target_idx]
    
    # Sort by similarity (descending)
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Print the results
    print(f"\nTarget customer: {descriptions[target_idx]}")
    print(f"\nTop {top_n} similar customers:")
    for i in range(min(top_n, len(similarity_scores))):
        idx, similarity = similarity_scores[i]
        print(f"Similarity: {similarity:.4f}")
        print(f"Customer: {descriptions[idx]}")
        print("-" * 50)


def main():
    """Main function to download model, load it, and embed data"""
    # Download model if needed
    model_path = download_model()
    
    # Load the model
    model = load_model()
    
    # Embed banking data
    embed_banking_data(model)


if __name__ == "__main__":
    main()