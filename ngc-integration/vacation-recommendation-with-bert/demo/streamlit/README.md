# How to Deploy and Use the Streamlit Web App

## 1. Configure for UI Mode

Before deployment, ensure the configuration is set for UI mode in the `configs/config.yaml` file:

```yaml
ui:
  mode: "streamlit" # Set to "streamlit" for Streamlit deployment
```

## 2. Register the Model

Run the model registration notebook to register your trained model with MLflow:

- Navigate to `notebooks/register-model.ipynb`
- Execute all cells to register the model in the MLflow Model Registry
- Note the model name and version for deployment

## 3. Deploy Using AI Studio

1. **Open the Deployment Tab** in Z by HP AI Studio
2. **Select Container Deployment** option
3. **Choose your registered model** from the MLflow Model Registry
4. **Configure deployment settings**:
   - Select appropriate compute resources
   - Set environment variables if needed
   - Configure networking options
5. **Deploy the container** - AI Studio will handle the containerization and deployment automatically
6. **Access the deployed app** through the provided URL

## 4. Using the Deployed Application

Once deployed:

- **Access the Streamlit interface** through the deployment URL provided by AI Studio
- **The MLflow endpoint** is automatically configured within the containerized environment
- **All model interactions** happen through the container-to-container communication
- **No manual setup** of Python environments or dependencies is required
