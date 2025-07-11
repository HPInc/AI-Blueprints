# Model Details  

## How the BERT Similarity Model Works  
- Utilizes **BERT embeddings** to measure similarity between user queries and dataset entries.  
- Converts text queries into vector representations and compares them against precomputed embeddings.  
- Leverages **MLflow** for efficient model logging, versioning, and retrieval.  

---

# API Endpoints  

## How the Frontend Interacts with the Backend  
- The system provides an `/invocations` API endpoint for processing user queries.  
- The frontend sends a request containing the user query to this endpoint.  
- The backend processes the request, identifies the most relevant results, and returns a structured response.  