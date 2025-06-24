# 🚫 Spam Detection with NLP

## 📚 Contents

- [🧠 Overview](#overview)
- [🗂 Project Structure](#project-structure)
- [⚙️ Setup](#setup)
- [🚀 Usage](#usage)
- [📞 Contact and Support](#contact-and-support)

---

# Overview

Simple text, specifically spam, classification using Natural Language Processing (NPL).

---

# Project Structure

```
├── docs/
│   └── swagger_UI_spam_detection_with_nlp.pdf           # Swagger screenshot
│   └── swagger_UI_spam_detection_with_nlp.png           # Swagger screenshot
├── notebooks
│   └── spam_detection_with_NLP.ipynb                    # Main notebook for the project             
├── README.md                                            # Project documentation
│
├── requirements.txt                                     # Dependency file for installing required packages
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
1. This experiment requires the **tutorial_data dataset** to run.
2. Download the dataset from `s3://dsp-demo-bucket/tutorial_data/` into an asset called **tutorial** and ensure that the AWS region is set to ```us-west-2```.

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
notebooks/spam_detection_with_NLP.ipynb
```

This will:

- Load and prepare the data
- Peform a Exploratory Data Analysis
- Preprocess the Text and Vectorize
- Train a Model
- Evaluate the Model
- Train Test Split
- Create a Data Pipeline
- Integrate MLflow

### 2 ▪ Deploy the Spam Detection with NLP SVM Service

- Go to **Deployments > New Service** in AI Studio.
- Name the service and select the registered model.
- Choose a model version and **GPU** it's **not necessary**.
- Choose the workspace.
- Start the deployment.
- Note: This is a local deployment running on your machine. As a result, if API processing takes more than a few minutes, it may return a timeout error. If you need to work with inputs that require longer processing times, we recommend using the provided notebook in the project files instead of accessing the API via Swagger or the web app UI.

### 3 ▪ Swagger / raw API

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
