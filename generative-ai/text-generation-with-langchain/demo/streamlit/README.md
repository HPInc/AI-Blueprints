# Streamlit UI Deployment Guide

## Overview

This Streamlit web application is designed for seamless deployment through Z by HP AI Studio's containerized deployment system. The application automatically integrates with your deployed MLflow model service without requiring manual setup or configuration.

## Deployment Instructions

### 1. Configure UI Mode

Set the UI mode in your project's configuration file:

**File:** `configs/config.yaml`

```yaml
ui:
  mode: "streamlit"
```

### 2. Register Your Model

Execute the model registration notebook:

- Open `notebooks/register-model.ipynb`
- Run all cells to log and register your model to MLflow
- This will automatically include the Streamlit UI as part of the deployment artifacts

### 3. Deploy via AI Studio

1. **Navigate to Deployments**: Go to **Deployments > New Service** in AI Studio
2. **Configure Service**:
   - Name your service
   - Select the registered model
   - Choose the model version
   - Configure GPU settings as needed
   - Select your workspace
3. **Deploy**: Click **Deploy** to start the service
4. **Launch**: Once deployed, click the play button to launch your service
5. **Access UI**: The Streamlit interface will be automatically available through the deployment URL

## Features

- **Automatic Integration**: No manual endpoint configuration required
- **Containerized Deployment**: Runs in an isolated, secure container environment
- **Seamless MLflow Connection**: Automatically connects to your deployed model service
- **Enterprise-Ready**: Built for Z by HP AI Studio's enterprise-grade platform

## Important Notes

- This is a **local deployment** running on your machine
- The Streamlit UI is automatically served as part of the containerized service
- No manual Python, Poetry, or dependency installation required
- The MLflow endpoint is automatically configured for container-to-container communication

## Troubleshooting

- **Service not starting**: Verify your model is properly registered in MLflow
- **UI not loading**: Check that the UI mode is set to "streamlit" in `configs/config.yaml`
- **Connection issues**: Ensure your workspace has sufficient resources allocated

For additional support, refer to the [AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview) or visit the [HP AI Creator Community](https://community.datascience.hp.com/).

---

Built with ❤️ using [**Z by HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).