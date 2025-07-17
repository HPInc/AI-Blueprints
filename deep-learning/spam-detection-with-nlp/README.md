# 🚫 Spam Detection with NLP

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-supported-orange.svg?logo=jupyter)
![NLP](https://img.shields.io/badge/NLP-used-brightgreen.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-used-f7931e.svg?logo=scikit-learn)

</div>

## 📚 Contents

- [🧠 Overview](#overview)
- [🗂 Project Structure](#project-structure)
- [⚙️ Setup](#setup)
- [🚀 Usage](#usage)
- [📞 Contact and Support](#contact-and-support)

---

# Overview

Text classification model capable of accurately distinguishing between spam and ham (non-spam) messages using Natural Language Processing (NLP) techniques and the Natural Language Toolkit (NLTK). The model is trained and evaluated using the spam_utf8.csv dataset, which contains labeled messages. Each entry in the dataset includes two columns: label, indicating whether the message is "spam" or "ham", and text, containing the actual content of the message.

---

# Project Structure

```
├── docs/
│   └── swagger-ui-spam-detection-with-nlp.pdf                    # Swagger screenshot
│   └── swagger-ui-spam-detection-with-nlp.png                    # Swagger screenshot
├── notebooks
│   └── register-model.ipynb                                      # Notebook for registering trained models to MLflow
│   └── run-workflow.ipynb                                        # Notebook for executing the pipeline using custom inputs and configurations             
├── README.md                                                     # Project documentation
│
├── requirements.txt                                              # Dependency file for installing required packages
```

---

# Setup

### 0 ▪ Minimum Hardware Requirements

Ensure your environment meets the minimum compute requirements for smooth image classification performance:

- **RAM**: 16 GB  
- **VRAM**: 4 GB  
- **GPU**: NVIDIA GPU

### 1 ▪ Create an AI Studio Project

- Create a new project in [Z by HP AI Studio](https://zdocs.datascience.hp.com/docs/aistudio/overview).

### 2 ▪ Set Up a Workspace

- Choose **Deep Learning** as the base image.

### 3 ▪ Download the Dataset

- Download the `tutorial_data dataset`

  - **Asset Name**: `tutorial` 
  - **Source**: `AWS S3`
  - **S3 URI**: `s3://dsp-demo-bucket/tutorial_data/`
  - **Resource Type**: `public`
  - **Bucket Region**: `us-west-2`

- Make sure that the model is in the datafabric folder inside your workspace. If the model does not appear after downloading, please restart your workspace.

### 4 ▪ Clone the Repositoryy

1. Clone the GitHub repository:  
   ```
   git clone https://github.com/HPInc/AI-Blueprints.git
   ```

2. Ensure all files are available after workspace creation.

---

# Usage

### 1 ▪ Run the Notebook

Execute the notebook inside the `notebooks` folder:

```bash
run-workflow.ipynb
```

This will:

- Load and prepare the data
- Peform a Exploratory Data Analysis
- Preprocess the Text and Vectorize
- Train a Model
- Evaluate the Model
- Train Test Split
- Create a Data Pipeline

### 2 ▪ Run the Notebook

Execute the notebook inside the `notebooks` folder:

```bash
register-model.ipynb
```

This will:

- Log Model to MLflow
- Fetch the Latest Model Version from MLflow
- Load the Model and Run Inference


### 3 ▪ Deploy the Spam Detection with NLP SVM Service

- Go to **Deployments > New Service** in AI Studio.
- Name the service and select the registered model.
- Choose a model version and **GPU** it's **not necessary**.
- Choose the workspace.
- Start the deployment.
- Note: This is a local deployment running on your machine. As a result, if API processing takes more than a few minutes, it may return a timeout error. If you need to work with inputs that require longer processing times, we recommend using the provided notebook in the project files instead of accessing the API via Swagger or the web app UI.

### 4 ▪ Swagger / raw API

Once deployed, access the **Swagger UI** via the Service URL.

Paste a payload like:

```
{
  "inputs": {
    "text": [
      "You have won a free ticket!"
    ]
  },
  "params": {}
}
```

And as response:

```
{
  "predictions": [
    "ham"
  ]
}

```
---

# Contact and Support  

- Issues: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting. 

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
