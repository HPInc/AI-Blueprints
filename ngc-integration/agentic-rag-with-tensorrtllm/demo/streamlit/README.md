````markdown
# How to Deploy and Use the Streamlit Web App

## 1. Configure for UI Mode

Before deployment, ensure the configuration is set for UI mode in the `configs/config.yaml` file:

```yaml
ui:
  mode: "streamlit" # Set to "streamlit" for Streamlit deployment
```

## 2. Register the Model

Run the model registration notebook to register your trained Agentic RAG model with MLflow:

- Navigate to `notebooks/register-model.ipynb`
- Execute all cells to register the model in the MLflow Model Registry
- This will create the Agentic RAG model with all necessary components (TensorRT-LLM, LangGraph, ChromaDB, etc.)
- Note the model name and version for deployment

## 3. Deploy Using AI Studio

1. **Open the Deployment Tab** in Z by HP AI Studio
2. **Select Container Deployment** option
3. **Choose your registered model** from the MLflow Model Registry (look for "Agentic_RAG_Model")
4. **Configure deployment settings**:
   - Select appropriate compute resources (GPU with at least 32GB VRAM recommended)
   - Set environment variables if needed
   - Configure networking options
   - Ensure port 5002 is exposed for the MLflow endpoint
5. **Deploy the container** - AI Studio will handle the containerization and deployment automatically
6. **Access the deployed app** through the provided URL

## 4. Using the Deployed Application

Once deployed, you can interact with the Agentic RAG system:

### Main Features:

- **Ask Questions**: Enter questions about AI Studio in the text area
- **Get Intelligent Answers**: The system uses a multi-step agentic workflow:
  - Checks query relevance
  - Looks up cached answers in memory
  - Rewrites queries for better retrieval
  - Retrieves relevant context from ChromaDB
  - Generates answers using TensorRT-LLM
  - Stores new answers in memory cache

### Example Queries:

- What is AI Studio?
- How to create a project in AI Studio?
- What are the technical requirements of AI Studio?
- How can I create a workspace in AI Studio?

### Optional Display Features:

- **Show Retrieved Context**: View the actual context chunks retrieved from the knowledge base
- **Show Metadata**: See additional information like whether the answer came from cache, query rewriting, etc.

## 5. Architecture Overview

The Agentic RAG system includes:

- **TensorRT-LLM**: GPU-accelerated inference with Llama-3.1-Nemotron-Nano model
- **LangGraph**: Orchestrates the multi-step agentic workflow
- **ChromaDB**: Vector database for context retrieval
- **SimpleKVMemory**: In-memory cache for faster repeated queries
- **MLflow**: Model registry and serving infrastructure

## 6. Local Development

To run the Streamlit app locally for development:

```bash
# Navigate to the demo directory
cd ngc-integration/agentic-rag-with-tensorrtllm/demo/streamlit

# Install dependencies
poetry install

# Run the Streamlit app
streamlit run main.py
```

**Note**: Make sure the MLflow model endpoint is running and accessible at `http://localhost:5002/invocations`

## 7. Troubleshooting

### Common Issues:

**Error connecting to MLflow endpoint:**
- Ensure the model is deployed and the service is running
- Verify the endpoint URL in `main.py` matches your deployment
- Check that port 5002 is accessible
- Confirm the model is registered in MLflow Model Registry

**GPU Memory Issues:**
- The TensorRT-LLM model requires at least 32GB VRAM
- Consider using a more powerful GPU or optimizing the model

**Slow Response Times:**
- First query may be slow as the model initializes
- Subsequent queries should be faster, especially cached ones
- Check GPU utilization and ensure TensorRT is properly optimized

## 8. System Requirements

### Minimum Hardware:
- **GPU**: NVIDIA GPU with at least 32GB VRAM
- **RAM**: ≥ 64GB system memory
- **Disk**: ≥ 32GB free space

### Software:
- Python 3.10+
- CUDA-compatible GPU drivers
- NeMo Framework 25.04 (for AI Studio deployment)

---

> Built with ❤️ using HP AI Studio.
````
