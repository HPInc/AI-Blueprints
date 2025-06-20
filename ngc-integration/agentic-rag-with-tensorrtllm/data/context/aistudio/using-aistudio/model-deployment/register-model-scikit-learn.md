---
title:  'Register a Model - Scikit-learn'
sidebar_position: 3
---
# How to Register a Model to MLflow: A Step-by-Step Guide

This guide outlines the steps necessary to register a model to the MLflow Model Registry, which provides an organized structure for model lifecycle management, including versioning, annotations, and staging or production-ready deployment. 

By following these steps, you can successfully register, manage, and deploy machine learning models using MLflow’s Model Registry. The Model Registry not only simplifies model management but also ensures traceability and consistency across versions, making it an indispensable tool for machine learning operations (MLOps).

Let’s get started with registering a model. 

## Step 0: Install Required Dependencies
To use MLflow and its model registry, you first need to install the MLflow library. AI Studio workspace images automatically do this for you. To re-install, start a workspace and open your terminal or command prompt and run the following command:

```bash
pip install --upgrade mlflow
```

This ensures that MLflow and all of its necessary dependencies are installed and up-to-date.

## Step 1: Train and Log the Model
Before registering a model, it must be logged using MLflow’s logging methods. For example, if you are using Scikit-learn, you can follow this process to embed hooks in your notebook:

1. **Start an MLflow Run**: This is a container for tracking parameters, metrics, and artifacts (like models).
2. **Train a Model**: Train your machine learning model using your preferred framework.
3. **Log the Model**: Use `mlflow.<model_flavor>.log_model()` to log your model to the tracking server.

    ```python
    import mlflow

    import mlflow.sklearn

    from sklearn.ensemble import RandomForestRegressor

    # Example: Train and log a RandomForest model
    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)

        # Log the model
        mlflow.sklearn.log_model(model, "random_forest_model")
    ```

In this case, the **RandomForestRegressor** model is trained and then logged to MLflow as a Scikit-learn model flavor.

## Step 2: Register the Model
Now that the model has been logged, it can be registered to the Model Registry.

MLflow offers two ways to register models:

1. **Register While Logging**: You can log and register the model simultaneously by specifying the `registered_model_name` parameter:

    ```python
    mlflow.sklearn.log_model(model, "random_forest_model", registered_model_name="RandomForestModel")
    ```

2. **Register After Logging**: If you have already logged a model and wish to register it afterward, use `mlflow.register_model()`. First, obtain the model’s URI and then register it:

    ```python
    model_uri = "runs:/<run-id>/random_forest_model"
    mlflow.register_model(model_uri, "RandomForestModel")
    ```

Both approaches add the model to the Model Registry, where it can be versioned, annotated, and transitioned between stages (e.g., Staging, Production).

## Step 3: View and Manage the Model
Once registered, you can view and manage your model from the MLflow UI or programmatically. To access the MLFlow UI, simply click the **Monitor** tab in your project.

- **Via the MLflow UI**: You can access the model by navigating to the "Models" section in the MLflow UI. Here, you can see all versions of the model, assign stages (e.g., Staging, Production), and add descriptions or annotations.
- **Programmatically**: You can interact with the Model Registry programmatically using the MLflow client API.

**Example of listing all versions of a model**:

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
versions = client.get_registered_model("RandomForestModel").latest_versions
for version in versions:
    print(f"Version: {version.version}, Stage: {version.current_stage}")
```

## Step 4: Transition Model Stages
MLflow provides a robust mechanism to manage model versions and their lifecycle. You can move models across different stages: **None**, **Staging**, **Production**, or **Archived**.

For example, to move a model version to Production:

```python
client.transition_model_version_stage(
    name="RandomForestModel",
    version=1,
    stage="Production"
)
```

This helps manage models that are ready for deployment and tracks their lifecycle status.

## Step 5: Load a Registered Model
Once the model has been registered, it can be loaded for inference using `mlflow.pyfunc.load_model()` or the specific flavor you logged. For example:

```python
model = mlflow.pyfunc.load_model(model_uri="models:/RandomForestModel/Production")
```
You can replace "Production" with "Staging" or specify a particular version.

**How to Save a Model as a Python Function**

MLflow provides the ability to log a Python function as a model using the `python_function` model flavor. This is a flexible format that allows any Python model or code to be registered and deployed. Here's how you can do it:

**Function-Based Model**

If you have a simple function that you want to log, you can use the `mlflow.pyfunc.log_model()` method. For example:

```python
import mlflow
import pandas as pd

# Define a simple predict function
def predict(model_input):
    return model_input.apply(lambda x: x * 2)

# Log the function as a model
with mlflow.start_run():
    mlflow.pyfunc.log_model("model", python_model=predict, pip_requirements=["pandas"])
    run_id = mlflow.active_run().info.run_id

# Load the model and perform inference
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
x_new = pd.Series([1, 2, 3])
prediction = model.predict(x_new)
print(prediction)
```

This logs the predict function as a model, which can then be loaded and used for inference.

**Class-Based Model**

If you need more flexibility or power, such as custom preprocessing or complex logic, you can create a class that implements the predict method and log it:

```python
import mlflow
import pandas as pd

class MyModel(mlflow.pyfunc.PythonModel):

    def predict(self, context, model_input):
        return [x * 2 for x in model_input]

# Log the class-based model
with mlflow.start_run():
    mlflow.pyfunc.log_model("model", python_model=MyModel(), pip_requirements=["pandas"])
    run_id = mlflow.active_run().info.run_id

# Load the model and perform inference
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
x_new = pd.Series([1, 2, 3])
print(model.predict(x_new))
```

The class-based approach allows you to define more complex models with custom logic.

**Loading and Using the Python Function Model**

After saving the model as a Python function (either function-based or class-based), you can load and use it just like any other MLflow model.

```python
# Load the Python function model
model = mlflow.pyfunc.load_model("runs:/<run-id>/model")

# Use it to predict on new data
result = model.predict(input_data)
```

This is particularly useful when integrating custom Python models into the MLflow ecosystem, allowing them to be deployed just like traditional machine learning models.

:::tip

**Once your models are registered to MLFlow, you can publish them for local inference using the deployment tab in your project.** 

:::

**Model Flavors Supported by MLflow**

MLflow supports a wide variety of model "flavors" or formats, such as:

- **Scikit-learn** (mlflow.sklearn)
- **TensorFlow** (mlflow.tensorflow)
- **PyTorch** (mlflow.pytorch)
- **XGBoost** (mlflow.xgboost)
- **Python Function** (mlflow.pyfunc) A generic format for Python models.

Each flavor has specific logging methods (e.g., `mlflow.sklearn.log_model()`) to log models into MLflow’s Model Registry. You can choose the flavor based on your framework to ensure compatibility during registration and inference.
