import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

def get_query_embedding(query, tokenizer, model):
    """
    Generate embedding for a search query using the same model used for the dataset.
    
    Args:
        query (str): The search query text
        tokenizer: The Hugging Face tokenizer
        model: The Hugging Face model
        
    Returns:
        numpy.ndarray: The embedding for the query
    """
    # Tokenize the query
    encoded_input = tokenizer(
        query, 
        padding=True, 
        truncation=True, 
        max_length=512,
        return_tensors='pt'
    )

    # Generate embedding
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Use mean pooling to get sentence embedding
    embedding = model_output.last_hidden_state.mean(dim=1)
    
    # Convert to numpy array and reshape to match the format of our stored embeddings
    return embedding[0].cpu().numpy().reshape(1, -1)

def vector_search(query, embeddings_csv, netflix_csv, top_n=3):
    """
    Perform vector search on the embeddings for a given query.
    
    Args:
        query (str): The search query
        embeddings_csv (str): Path to the embeddings CSV
        netflix_csv (str): Path to the Netflix reviews CSV
        top_n (int): Number of top results to return
        
    Returns:
        pandas.DataFrame: DataFrame with the top matching results
    """
    print(f"Loading embeddings from {embeddings_csv}")
    embeddings_df = pd.read_csv(embeddings_csv)
    
    print(f"Loading Netflix data from {netflix_csv}")
    netflix_df = pd.read_csv(netflix_csv)
    
    if len(embeddings_df) != len(netflix_df):
        print(f"Warning: Embeddings count ({len(embeddings_df)}) doesn't match Netflix data count ({len(netflix_df)})")
    
    # Load the Hugging Face model
    print("Loading Hugging Face model for query embedding")
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Generate embedding for the query
    print(f"Generating embedding for query: '{query}'")
    query_embedding = get_query_embedding(query, tokenizer, model)
    
    # Convert dataframe to numpy array for faster computation
    embeddings = embeddings_df.values
    
    # Calculate cosine similarity
    print("Calculating similarities with all embeddings")
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get indices of top N most similar embeddings
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    # Create a DataFrame with the results
    results = netflix_df.iloc[top_indices].copy()
    
    # Add similarity scores
    results['similarity_score'] = similarities[top_indices]
    
    return results

if __name__ == "__main__":
    # File paths
    embeddings_csv = "data/embedded.csv"
    netflix_csv = "data/netflix_reviews.csv"
    
    # Search query
    query = "action shows"
    
    # Perform the search
    results = vector_search(query, embeddings_csv, netflix_csv, top_n=3)
    
    # Display results
    print("\nTop Results for Query:", query)
    print("=" * 80)
    
    # Create a nicely formatted display of results
    for i, (_, row) in enumerate(results.iterrows(), 1):
        print(f"{i}. {row.get('title', 'Unknown')} ({row.get('release_year', 'Unknown')})")
        
        # Show genre if available
        if 'listed_in' in row and not pd.isna(row['listed_in']):
            print(f"   Genre: {row['listed_in']}")
            
        # Show description if available
        if 'description' in row and not pd.isna(row['description']):
            # Truncate description if too long
            desc = row['description']
            if len(desc) > 150:
                desc = desc[:147] + "..."
            print(f"   Description: {desc}")
            
        # Show similarity score
        print(f"   Similarity: {row['similarity_score']:.4f}")
        print("-" * 80)
