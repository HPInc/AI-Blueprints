# 🌷 Classification with SVM

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-supported-orange.svg?logo=jupyter)
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

This project is a simple **classification** experiment focused on predicting species of **Iris flowers**.

It runs on the **Data Science Workspace**, demonstrating basic supervised learning techniques for multi-class classification tasks.

---

# Project Structure

```
├── docs/
│   └── swagger_UI_classification_with_svm.pdf                 # Swagger screenshot
│   └── swagger_UI_classification_with_svm.png                 # Swagger screenshot
├── notebooks
│   └── iris_flowers_classification_with_svm.ipynb             # Main notebook for the project              
├── README.md                                                  # Project documentation
```

---

# Setup

### 0 ▪ Minimum Hardware Requirements

Ensure your environment meets the minimum compute requirements for smooth image classification performance:

- **RAM**: 4 GB  

### 1 ▪ Create an AI Studio Project

- Create a new project in [Z by HP AI Studio](https://zdocs.datascience.hp.com/docs/aistudio/overview).

### 2 ▪ Set Up a Workspace

- Choose **Data Science** as the base image.

### 3 ▪ Clone the Repository

1. Clone the GitHub repository:  
   ```
   git clone https://github.com/HPInc/AI-Blueprints.git
   ```

2. Ensure all files are available after workspace creation.

---

# Usage

### 1 ▪ Run the Notebook

Execute the run-workflow notebook first inside the `notebooks` folder first.

```bash
notebooks/eun-workllow.ipynb
```

Execute the register-model notebook second inside the `notebooks` folder:

```bash
notebooks/register-model.ipynb
```

This will:

- Load and prepare the data
- Summarize the Dataset, doing a overview.
- Visualize the data
- Build the Model and Measure Model.
- Integrate MLflow

### 2 ▪ Deploy the Iris flowers classification with SVM and LDA Service

- Go to **Deployments > New Service** in AI Studio.
- Name the service and select the registered model.
- Choose a model version and **GPU** is **not necessary**.
- Choose the workspace.
- Start the deployment.
- Note: This is a local deployment running on your machine. As a result, if API processing takes more than a few minutes, it may return a timeout error. If you need to work with inputs that require longer processing times, we recommend using the provided notebook in the project files instead of accessing the API via Swagger or the web app UI.

### 3 ▪ Swagger / raw API

Once deployed, access the **Swagger UI** via the Service URL.


Paste a payload like:

```
{
  "inputs": {
    "sepal-length": [
      5.1
    ],
    "sepal-width": [
      3.5
    ],
    "petal-length": [
      1.4
    ],
    "petal-width": [
      0.2
    ]
  },
  "params": {}
}
```
And as response:

```
{
  "predictions": [
    "Iris-setosa"
  ]
}

```
---

# Contact and Support

- **Troubleshooting:** Refer to the [**Troubleshooting**](https://github.com/HPInc/AI-Blueprints/tree/main?tab=readme-ov-file#troubleshooting) section of the main README in our public AI-Blueprints GitHub repo for solutions to common issues.

- **Issues & Bugs:** Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- **Docs:** [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

- **Community:** Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://hp.com/ai-studio).
