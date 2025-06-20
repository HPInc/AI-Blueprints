---
title:  'Training a PyTorch Model and Logging with MLFlow'
sidebar_position: 4
---
# Training a PyTorch Model and Logging with MLFlow

In this guide, we'll walk through the entire process of training a breast cancer classification model using PyTorch and logging the process with MLflow in the AI Studio environment. We will go step-by-step through data exploration, model training, and tracking model performance using MLflow. This tutorial is designed to be beginner-friendly, and we'll cover each step in detail to help you build and deploy your model effectively. 

## Step 1: Install Required Libraries 

First, ensure you have all the necessary dependencies installed. If you are working in AI Studio, many of these dependencies are already pre-installed, including MLflow. This saves setup time and helps you get started faster. If you are using a custom workspace, you can install the dependencies manually: 

```python
import pandas as pd 
import numpy as np 
import mlflow.pytorch 
import seaborn as sns 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler 
```

To install any missing dependencies, use: 

```sh
pip install mlflow torch pandas seaborn 
```

## Step 2: Load and Explore the Data 

In this tutorial, we use a breast cancer classification dataset. The dataset contains features about breast tumors that we will use to predict whether a tumor is benign or malignant. 

Load the dataset into a pandas DataFrame and explore the data to understand its structure: 

```python
df = pd.read_csv('../data/cancer_classification.csv') 
df.info() 
print(df.describe().transpose())
``` 

### Exploratory Data Analysis (EDA) 

Understanding the data distribution is crucial before building a model. We'll perform a basic EDA to visualize the class distribution and relationships between features. 

**Countplot of Diagnosis** 

```python
sns.countplot(x='diagnosis', data=df) 
plt.show() 
```

Here, we are plotting the count of benign and malignant diagnoses to see if the classes are balanced. 

![Benign and Malignant Diagnoses Plot](/img/benign_malignant_plot.png)

**Correlation Heatmap** 

Visualize feature correlations to understand how different features are related: 

```python
df['diagnosis'] = df['diagnosis'].apply(lambda x: 0 if x == 'benign' else 1) 
sns.heatmap(df.corr(), annot=True) 
plt.show() 
```

This helps identify which features might be more important for predicting the diagnosis. 

![Correlation Heatmap](/img/correlation_heatmap.png)

## Step 3: Train-Test Split 

Next, we split the dataset into training and testing sets. This will help us evaluate our model's performance on unseen data. 

```python
X = df.drop('diagnosis', axis=1).values 
y = df['diagnosis'].values 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101) 
```

## Step 4: Data Scaling 

Neural networks tend to perform better when the input data is scaled. We use the MinMaxScaler to normalize the feature values between 0 and 1: 

```python
scaler = MinMaxScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 
```

## Step 5: Define the PyTorch Model 

We define a simple neural network using PyTorch's nn.Module. The model contains three fully connected layers with ReLU activations and dropout layers to prevent overfitting. 

```python
class BreastCancerModel(nn.Module): 
    def __init__(self): 
        super(BreastCancerModel, self).__init__() 
        self.fc1 = nn.Linear(X_train.shape[1], 30) 
        self.dropout1 = nn.Dropout(0.3) 
        self.fc2 = nn.Linear(30, 15) 
        self.dropout2 = nn.Dropout(0.3) 
        self.fc3 = nn.Linear(15, 1) 
 
    def forward(self, x): 
        x = torch.relu(self.fc1(x)) 
        x = self.dropout1(x) 
        x = torch.relu(self.fc2(x)) 
        x = self.dropout2(x) 
        x = self.fc3(x) 
        return x 
 
model = BreastCancerModel() 
```

## Step 6: Train the Model and Track with MLflow 

We use **BCEWithLogitsLoss** as our loss function, which combines a sigmoid layer with binary cross-entropy loss for numerical stability. We also use the Adam optimizer with a learning rate of 0.0001 to train our model. 

MLflow is used to track training metrics like loss, as well as to log the model for future use. This allows you to manage different versions and keep track of model performance. 

```python
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.0001) 
 
mlflow.pytorch.autolog() 
 
with mlflow.start_run(): 
    num_epochs = 50 
    train_losses = [] 
    val_losses = [] 
     
    for epoch in range(num_epochs): 
        model.train() 
        inputs = torch.tensor(X_train, dtype=torch.float32) 
        targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1) 
 
        optimizer.zero_grad() 
        outputs = model(inputs) 
        loss = criterion(outputs, targets) 
        loss.backward() 
        optimizer.step() 
 
        # Validation 
        model.eval() 
        with torch.no_grad(): 
            val_inputs = torch.tensor(X_test, dtype=torch.float32) 
            val_targets = torch.tensor(y_test, dtype=torch.float32).view(-1, 1) 
            val_outputs = model(val_inputs) 
            val_loss = criterion(val_outputs, val_targets) 
 
        train_losses.append(loss.item()) 
        val_losses.append(val_loss.item()) 
 
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}') 
 
    # Plot the Training and Validation Loss 
    plt.figure(figsize=(10, 5)) 
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss') 
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss') 
    plt.xlabel('Epochs') 
    plt.ylabel('Loss') 
    plt.legend() 
    plt.title('Training and Validation Loss Over Epochs') 
    plt.show() 
```

![Training and Validation Loss](/img/training_and_validation_loss.png)

## Step 7: Evaluate the Model 

After training the model, it's essential to evaluate its performance using metrics such as precision, recall, and F1-score. We generate predictions on the test set and assess the model using **classification_report** and **confusion_matrix** from sklearn. 

```python
model.eval() 
with torch.no_grad(): 
    val_inputs = torch.tensor(X_test, dtype=torch.float32) 
    predictions = torch.sigmoid(model(val_inputs)).numpy() 
    predictions_binary = [1 if p > 0.5 else 0 for p in predictions] 
 
from sklearn.metrics import classification_report, confusion_matrix 
 
print(classification_report(y_test, predictions_binary)) 
print(confusion_matrix(y_test, predictions_binary))
``` 

The classification report provides precision, recall, and F1-score for each class, which helps assess the model's ability to correctly classify benign and malignant cases. The confusion matrix gives a detailed view of true positives, false positives, true negatives, and false negatives. 

![Model Evaluation](/img/model_evaluation.png)

**Monitor Registered Model in MLflow** 

Once the model is registered in MLflow, you can go to the 'Monitor' tab in AI Studio to view its details, including versioning, performance metrics, and stage transitions in the MLflow Model Registry. This helps track the model's lifecycle from experimentation to production, ensuring easy deployment and reproducibility. 

![Registered Model in MLFlow](/img/registered_model_mlflow.png)

## Summary 

In this guide, we walked through the entire workflow for building a breast cancer classification model using PyTorch, and logging and managing the model lifecycle using MLflow within the AI Studio environment. We covered data exploration, model training, loss tracking, and evaluation. 

Using AI Studio's pre-configured environment makes it easy to set up MLflow and begin tracking your models, enabling you to focus on developing and improving the model itself rather than dealing with configuration challenges. Feel free to extend this workflow to more complex models or datasets to meet your specific use cases. 