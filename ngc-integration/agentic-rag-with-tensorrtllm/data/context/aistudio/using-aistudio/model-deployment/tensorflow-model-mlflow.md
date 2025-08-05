---
title:  'Training a TensorFlow Model and Publishing via MLflow'
sidebar_position: 5
---
# Training a TensorFlow Model and Publishing via MLflow

In this example, we will demonstrate how to train a model with TensorFlow and register it with MLflow in AI Studio. The goal is to build a breast cancer classification model, track experiments, and publish the results. We will use a public dataset to demonstrate each step of the process, starting with data exploration, then training, and finally using MLflow for registration and publishing. 

## Step 1: Install Required Libraries 

If you are using a custom workspace, ensure that all necessary requirements are installed by either adding them to a requirements file or manually installing them with these commands: 

```python
import pandas as pd 
import numpy as np 
import mlflow.tensorflow
```

You will need mlflow, tensorflow, pandas, seaborn, and other related libraries. If not already installed, you can use: 

```sh
pip install mlflow tensorflow pandas seaborn
``` 

You can also use predefined workspaces that already have many of these dependencies installed, including MLflow, to help you get started quicker. 

## Step 2: Load and Explore the Data 

We will begin by loading the breast cancer dataset: 

```
df = pd.read_csv('../data/cancer_classification.csv') 
```

Check the dataset's information to understand the feature types and view missing values with the following commands: 

```python
df.info() 
<class 'pandas.core.frame.DataFrame'> 
RangeIndex: 569 entries, 0 to 568 
Data columns (total 31 columns): 
 #   Column                   Non-Null Count  Dtype   
---  ------                   --------------  -----   
 0   mean radius              569 non-null    float64 
 1   mean texture             569 non-null    float64 
 2   mean perimeter           569 non-null    float64 
 3   mean area                569 non-null    float64 
 4   mean smoothness          569 non-null    float64 
 5   mean compactness         569 non-null    float64 
 6   mean concavity           569 non-null    float64 
 7   mean concave points      569 non-null    float64 
 8   mean symmetry            569 non-null    float64 
 9   mean fractal dimension   569 non-null    float64 
 10  radius error             569 non-null    float64 
 11  texture error            569 non-null    float64 
 12  perimeter error          569 non-null    float64 
 13  area error               569 non-null    float64 
 14  smoothness error         569 non-null    float64 
 15  compactness error        569 non-null    float64 
 16  concavity error          569 non-null    float64 
 17  concave points error     569 non-null    float64 
 18  symmetry error           569 non-null    float64 
 19  fractal dimension error  569 non-null    float64 
 20  worst radius             569 non-null    float64 
 21  worst texture            569 non-null    float64 
 22  worst perimeter          569 non-null    float64 
 23  worst area               569 non-null    float64 
 24  worst smoothness         569 non-null    float64 
 25  worst compactness        569 non-null    float64 
 26  worst concavity          569 non-null    float64 
 27  worst concave points     569 non-null    float64 
 28  worst symmetry           569 non-null    float64 
 29  worst fractal dimension  569 non-null    float64 
 30  benign_0__mal_1          569 non-null    int64   
dtypes: float64(30), int64(1) 
memory usage: 137.9 KB 
```

To view descriptive statistics for a summary of each feature: 

```python
df.describe().transpose() 
```
![Descriptive Statistics](/img/descriptive_statistics_each_feature.png)

## Step 3: Exploratory Data Analysis (EDA) 

Understanding the data distribution is an important step. We will use seaborn and matplotlib commands to explore the example dataset. 

### Countplot of Target Variable 

Plotting the count of benign and malignant cases helps visualize class distribution: 

```python
import seaborn as sns 
import matplotlib.pyplot as plt 
sns.countplot(x='benign_0__mal_1', data=df) 
plt.show() 
```

### Correlation Heatmap 

Understanding feature correlation is key to building an effective model: 

```python
sns.heatmap(df.corr(), annot=True) 
plt.show() 
```

![Correlation Heatmap Tensorflow](/img/correlation_heatmap_tensorflow.png)

## Step 4: Train-Test Split

Before we train the model, split the dataset into training and testing sets: 

```python
from sklearn.model_selection import train_test_split 
 
X = df.drop('benign_0__mal_1', axis=1).values 
y = df['benign_0__mal_1'].values 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101) 
```

## Step 5: Data Scaling

Neural networks perform better with scaled data, so we’ll scale features using MinMaxScaler: 

```python
from sklearn.preprocessing import MinMaxScaler 
 
scaler = MinMaxScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 
```

## Step 6: Train a TensorFlow Model 

To create and train a simple TensorFlow neural network model: 

```python
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout 
 
model = Sequential([ 
    Dense(30, activation='relu'), 
    Dropout(0.5),  # Add dropout to prevent overfitting 
    Dense(15, activation='relu'), 
    Dropout(0.5),  # Add dropout to prevent overfitting 
    Dense(1, activation='sigmoid') 
]) 
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
 
# Use EarlyStopping to prevent overtraining 
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) 
 
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping])
```

To avoid overfitting, we use techniques like Dropout layers, which randomly deactivate neurons during training to help the model with generalization. Early Stopping is used to stop training when validation loss stops improving, preventing the model from overtraining and saving you time and resources. 

For the purposes of this example, we’ll continue on without using those overfitting reduction techniques and move on to how to log and register themodel with MLFlow.

## Step 7: Log the Model with MLflow

We can use MLflow to log our TensorFlow model. This allows us to keep track of different versions and monitor performance: 

```python
mlflow.tensorflow.autolog() 
 
with mlflow.start_run(): 
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test)) 
```

Using mlflow.autolog() automatically logs metrics, parameters, and the model itself. 

## Step 8: Register the Model in MLflow

Once you log the model, you can register it to the MLflow Model Registry. This lets you version the model and track its lifecycle. 

### To register while logging: 

Use the registered_model_name parameter: 

```python
mlflow.tensorflow.log_model(model, "breast_cancer_model", registered_model_name="BreastCancerModel") 
```

### To register after logging: 

Use the following parameter: 

```python
model_uri = "runs:/<run-id>/breast_cancer_model" 
mlflow.register_model(model_uri, "BreastCancerModel") 
```

You can transition the model between stages like Staging and Production for use in deployment. Once registered, you can also publish the model in the Published Services tab for easy access and integration. 

## Step 9: Publish and Monitor the Model

After registering your model in MLflow, navigate to the Monitor tab in AI Studio to view its details, including versioning, performance metrics, and transition stages. This feature allows you to track your model's lifecycle from experimentation through to production and facilitates seamless deployment and monitoring. 

![Registered Model MLFlow Tensorflow](/img/registered_model_mlflow_tensorflow.png)

## Summary

In this example, we covered the entire workflow for training a breast cancer classification model using TensorFlow, logging it with MLflow, and registering it for easy management and deployment. Using MLflow’s capabilities to track and register models ensures efficient model management and deployment across environments. 

AI Studio provides a streamlined environment for working with MLflow, TensorFlow and many other machine learning tools, making it easy to track experiments, manage models, and publish them for use in production systems. 

For more information on MLflow and its capabilities, you can visit the official [MLflow documentation.](https://mlflow.org/docs/latest/index.html)

Feel free to extend this workflow to more complex models or datasets to meet your specific use cases. 