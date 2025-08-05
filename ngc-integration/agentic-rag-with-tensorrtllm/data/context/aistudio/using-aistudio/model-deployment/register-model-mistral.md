---
title:  'Register a Model - Mistral'
sidebar_position: 2
---

# How to Register a LLM Model to MLflow: A Step-by-Step Guide

HP AI Studio integrates an IDE, MLFlow, frameworks, and Swagger so you don’t have to. To register a locally trained LLM model, such as Mistral to MLFlow, ensure that you have downloaded, trained, and tuned your model, and you are ready to save your model to the MLFlow registry.

The benefits for doing this includes model versioning, evaluation, and deployment to Swagger. Follow these steps to register your Mistral model to MLFlow.

## Download the model locally

Download the model you will be using to your workstation, if you have not already done so. The following open source **LLM models** are popular choices; **Mistral**, **Llama**, **Gemma**, **Qwen**, and **DeepSeek**. You can distill, fine-tune, create a RAG, or prompt engineer on these models in AI Studio.

For this example, we will download and use **Mistral 7B (mistral-7b-v0.1.Q5_K_M.gguf)**.

## NVIDIA’s NGC Catalog

AI Studio has NVIDIA’s NGC Catalog of GPU optimized assets integrated, so you can search, find, get, and use models and containers.

For this example, we will use the **NeMO Framework container**, which you can select from the image catalog in AI Studio when creating your project workspace.

![NeMo Framwork Container](/img/NeMo_framework.png)

## Register the model

With your model and workspace configured, import the correct libraries to properly register your model to MLFLow. The following code creates a class to register a formatted model and point to model and web app (demo) artifacts.

You can copy and paste the following code to test how to register the Mistral gguf model to MLFlow. The only thing to change is the ‘model_path’. Make sure to copy the model path and model name from your datafabric folder and place it after **‘/home/joyvan/’**. It will look like **‘/home/jovyan/datafabric/mistral-7b-engine/mistral-7b-v0.1.Q5_K_M.gguf’**.

:::note 

Ensure your model is in your project workspace **‘Asset’** page. This will happen automatically if you specify the folder location where your Mistral model is stored locally. If you do not see it in the Asset page as downloaded, download it. 

:::

If you do not see it in your Jupyter folder structure under **‘datafabric’**, close the workspace, go to the Asset page in your project and add the model by pointing to the location it is locally stored.

```python
import os
import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, ParamSchema, ParamSpec

class AIStudioChatbotService(PythonModel):
    @classmethod
    def log_model(cls, model_folder=None, demo_folder="demo"):
        # Ensure the demo folder exists
        if demo_folder and not os.path.exists(demo_folder):
            os.makedirs(demo_folder, exist_ok=True)

        # Define input schema for the model
        input_schema = Schema([
            ColSpec("string", "query"),
            ColSpec("string", "prompt"),
            ColSpec("string", "document")
        ])
        
        # Define output schema for the model
        output_schema = Schema([
            ColSpec("string", "chunks"),
            ColSpec("string", "history"),
            ColSpec("string", "prompt"),
            ColSpec("string", "output"),
            ColSpec("boolean", "success")
        ])
        
        # Define parameters schema for additional settings
        param_schema = ParamSchema([
            ParamSpec("add_pdf", "boolean", False),
            ParamSpec("get_prompt", "boolean", False),
            ParamSpec("set_prompt", "boolean", False),
            ParamSpec("reset_history", "boolean", False)
        ])
        
        # Combine schemas into a model signature
        signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=param_schema)

        # Define model artifacts
        artifacts = {"demo": demo_folder}
        if model_folder:
            artifacts["models"] = model_folder

        # Log the model in MLflow
        mlflow.pyfunc.log_model(
            artifact_path="aistudio_chatbot_service",
            python_model=cls(),
            artifacts=artifacts,
            signature=signature,
            pip_requirements=[
                "pyyaml",
                "tokenizers==0.20.3",
                "httpx==0.27.2",
            ]
        )
        print("Model and artifacts successfully registered in MLflow.")

# Initialize the MLflow experiment
print("Initializing experiment in MLflow.")
mlflow.set_experiment("AIStudioChatbot_Service")

# Define required paths
model_folder = "/home/jovyan/datafabric/mistral-7b-engine/mistral-7b-v0.1.Q5_K_M.gguf"
demo_folder = "demo"   

# Ensure required directories exist before proceeding
if demo_folder and not os.path.exists(demo_folder):
    os.makedirs(demo_folder, exist_ok=True)

# Start an MLflow run and log the model
with mlflow.start_run(run_name="AIStudioChatbot_Service_Run") as run:
    AIStudioChatbotService.log_model(
        demo_folder=demo_folder,
        model_folder=model_folder
    )
    
    # Register the model in MLflow
    model_uri = f"runs:/{run.info.run_id}/aistudio_chatbot_service"
    mlflow.register_model(
        model_uri=model_uri,
        name="Mistral_Chatbot",
    )
    print(f"Registered model with execution ID: {run.info.run_id}")
    print(f"Model registered successfully. Run ID: {run.info.run_id}")

}
```


## Check in MLFlow

After running the code, navigate to MLFlow by clicking on the Monitor page. You will see your run. Click on **‘Model’** to see your registered model and version.

![Mistral MLFlow](/img/mistral_mlflow.png)

## Deploy your model to Swagger on localhost

With your model successfully registered in MLFlow in your AI Studio project, you are ready to deploy the model!

To deploy the model, click on the **Deployments** page, then **‘+New Service’**. Follow the prompts. be sure to select the same workspace you used to register the model, and click Deploy. Then simply click the play button under Action. Once the container is created and deployed, AI Studio will provide an endpoint you can click on to open Swagger.

## Interrogate your model through Swagger APIs

If you created a web app and saved page files in the demo folder (e.g. index.html or index.js), you will see a link to navigate to those at the top of Swagger. If you did not, you can click **‘Try it out’** to interrogate your locally hosted model.

With your model registered on MLFlow and hosted on localhost, you can access the endpoint AI Studio provides to interrogate the hosted model locally through the Swagger UI or a locally hosted web app.

![AI Studio Published Service](/img/aistudio_publishedservice.png)
