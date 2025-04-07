# 03_deployment.py
import os
import json
import torch
import numpy as np
import pandas as pd
from tabulate import tabulate
import mlflow
import mlflow.pyfunc

from mlflow import MlflowClient
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, TensorSpec, ParamSchema, ParamSpec
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModel

# Model settings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_model():
    """Load the HuggingFace model for embedding"""
    print(f"Loading HuggingFace model: {MODEL_NAME}")
    
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    
    return tokenizer, model, device

# 🏗️ Defining the Netflix Similarity Model Class
class NetflixSimilarityModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Load precomputed embeddings, Netflix reviews data, and HuggingFace model.
        """
        # Load precomputed embeddings
        self.embeddings_df = pd.read_csv(context.artifacts['embeddings_path'])
        
        # Load Netflix reviews corpus
        self.netflix_df = pd.read_csv(context.artifacts['netflix_reviews_path'])
        
        # Print diagnostics about the loaded data
        print(f"Loaded embeddings shape: {self.embeddings_df.shape}")
        print(f"Loaded Netflix reviews shape: {self.netflix_df.shape}")
        
        # Convert embeddings to numpy array for faster similarity computation
        self.embeddings = self.embeddings_df.values
        
        # Load HuggingFace model
        self.model_name = MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set device to GPU if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained model
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        print(f"HuggingFace model '{self.model_name}' loaded successfully")
    
    def generate_query_embedding(self, query):
        """Generate embedding for the input query text using HuggingFace model"""
        self.model.eval()  # Set model to evaluation mode
        
        # Tokenize the input query and move tensors to the selected device
        encoded_input = self.tokenizer(
            query, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=512
        )
        encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}
        
        # Get the model's output embedding
        with torch.no_grad():
            output = self.model(**encoded_input)
        
        # Use mean pooling to get sentence embedding
        embedding = output.last_hidden_state.mean(dim=1)
        
        # Return the embedding as a NumPy array
        return embedding.cpu().numpy()
    
    def predict(self, context, model_input, params):
        """
        Find similar Netflix content based on semantic similarity to the query text.
        Format results to be compatible with the HTML interface.
        """
        # Extract the query string from model input
        query = model_input["query"][0]
        print(f"Processing query: '{query}'")
        
        # Extract parameters
        top_n = params.get("top_n", 5) if params else 5
        
        # Make sure top_n isn't larger than our dataset
        top_n = min(top_n, len(self.netflix_df))
        
        # Generate embedding for the query text
        query_embedding = self.generate_query_embedding(query)
        
        # Compute cosine similarity between query and all embeddings
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get indices of top N most similar results
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        # Format results for the HTML interface
        predictions = []
        for idx in top_indices:
            show = self.netflix_df.iloc[idx]
            
            # Format Netflix content as a recommendation
            content = f"{show.get('title', 'Unknown')}"
            
            # Add genre if available
            if 'listed_in' in show and not pd.isna(show['listed_in']):
                content += f" - {show['listed_in']}"
                
            # Add description if available
            if 'description' in show and not pd.isna(show['description']):
                content += f": {show['description']}"
            
            # Format result to match HTML expectations
            result = {
                'Content': content,  
                'Similarity': float(similarities[idx])
            }
            
            predictions.append(result)
        
        # Return predictions as expected by HTML interface
        return {"predictions": predictions}
    
    @classmethod
    def log_model(cls, model_name, embeddings_path, netflix_reviews_path, demo_dir=None):
        """
        Logs the model to MLflow with appropriate artifacts and schema.
        """
        # Check if the files exist
        for path in [embeddings_path, netflix_reviews_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
        
        # Print file sizes for information
        emb_size = os.path.getsize(embeddings_path) / (1024 * 1024)
        netflix_size = os.path.getsize(netflix_reviews_path) / (1024 * 1024)
        
        print(f"Embeddings file size: {emb_size:.2f} MB")
        print(f"Netflix reviews file size: {netflix_size:.2f} MB")
        
        # Simple input schema - just accepting a query string
        input_schema = Schema([ColSpec("string", "query")])
        
        # Output schema now matches the HTML expectation structure
        output_schema = Schema([
            TensorSpec(np.dtype("object"), (-1,), "predictions")
        ])
        
        # Parameters schema - include show_score to match HTML interface
        params_schema = ParamSchema([
            ParamSpec("top_n", "integer", 5),
            ParamSpec("show_score", "boolean", True)
        ])
        
        # Define model signature
        signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=params_schema)
        
        # Define necessary package requirements
        requirements = [
            "scikit-learn",
            "pandas",
            "numpy",
            "tabulate",
            "torch",
            "transformers"
        ]
        
        # Define artifacts dictionary
        artifacts = {
            "embeddings_path": embeddings_path,
            "netflix_reviews_path": netflix_reviews_path
        }
        
        # Add demo directory to artifacts if provided and exists
        if demo_dir and os.path.exists(demo_dir):
            artifacts["demo"] = demo_dir
            
        # Define metadata with demo template if demo directory is provided and has index.html
        metadata = {}
        if demo_dir and os.path.exists(os.path.join(demo_dir, "index.html")):
            metadata["demo_template"] = "demo/index.html"
        
        # Log the model in MLflow
        mlflow.pyfunc.log_model(
            model_name,
            python_model=cls(),
            artifacts=artifacts,
            signature=signature,
            pip_requirements=requirements,
            metadata=metadata
        )

# 📜 Logging Model to MLflow
def log_model_to_mlflow():
    # Set the MLflow experiment name
    mlflow.set_experiment(experiment_name="Netflix HuggingFace Similarity")
    
    # Check if demo directory exists and has index.html
    demo_dir = "demo"
    index_html_path = os.path.join(demo_dir, "index.html")
    
    if os.path.exists(index_html_path):
        print(f"Found UI at {index_html_path}, will include in model deployment")
    else:
        print(f"Warning: UI file not found at {index_html_path}")
        os.makedirs(demo_dir, exist_ok=True)
        print("Creating demo directory...")

    # Start an MLflow run
    with mlflow.start_run(run_name="Netflix_HuggingFace_Run") as run:
        # Print the artifact URI for reference
        print(f"Run's Artifact URI: {run.info.artifact_uri}")
        
        # Log the Netflix similarity model to MLflow
        NetflixSimilarityModel.log_model(
            model_name="Netflix_HuggingFace_Similarity",
            embeddings_path="data/embedded.csv",
            netflix_reviews_path="data/netflix_reviews.csv",
            demo_dir=demo_dir if os.path.exists(demo_dir) else None
        )

        # Register the logged model in MLflow Model Registry
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/Netflix_HuggingFace_Similarity", 
            name="Netflix_HuggingFace_Similarity"
        )
        
        return run.info.run_id

# 📦 Function to load the model and run inference
def get_similar_content(query, run_id=None, top_n=5):
    """
    Get similar Netflix content for a given query.
    
    Args:
        query (str): The search query
        run_id (str, optional): MLflow run ID. If None, uses the latest model version
        top_n (int): Number of results to return
        
    Returns:
        DataFrame: Similar Netflix content
    """
    if run_id:
        # Load model from specific run
        model_uri = f"runs:/{run_id}/Netflix_HuggingFace_Similarity"
    else:
        # Get latest model version
        client = MlflowClient()
        model_metadata = client.get_latest_versions("Netflix_HuggingFace_Similarity", stages=["None"])
        latest_model_version = model_metadata[0].version
        model_uri = f"models:/Netflix_HuggingFace_Similarity/{latest_model_version}"
    
    # Load the model
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Prepare simple input data
    input_data = {"query": [query]}
    
    # Run inference with parameters
    result = model.predict(input_data, params={"top_n": top_n})
    
    # Extract predictions array from the result
    predictions = result.get("predictions", [])
    
    # Convert to DataFrame for better display
    return pd.DataFrame(predictions)

# 📊 Demo: Find similar Netflix content
def run_demo():
    # Log model to MLflow
    run_id = log_model_to_mlflow()
    
    if not run_id:
        print("Model logging failed.")
        return
    
    # Use a simple query
    query = "Show me something like Stranger Things"
    
    try:
        # Get similar content based on the query text
        similar_content = get_similar_content(
            query=query,
            run_id=run_id,
            top_n=5
        )
        
        # Display results
        print(f"\nQuery: {query}")
        print("\nTop similar Netflix content:")
        print(tabulate(similar_content, headers='keys', tablefmt='fancy_grid', showindex=False))
    
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Displaying sample of the Netflix reviews data instead:")
        netflix_df = pd.read_csv("data/netflix_reviews.csv", nrows=5)
        print(tabulate(netflix_df, headers='keys', tablefmt='fancy_grid', showindex=False))

# Run the demo if executed directly
if __name__ == "__main__":
    run_demo()