# 📈 Stock Analysis with Pandas and cuDF  

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-supported-orange.svg?logo=jupyter)
![cuDF](https://img.shields.io/badge/cuDF-GPU%20accelerated-76B900.svg?logo=nvidia)
![RAPIDS](https://img.shields.io/badge/RAPIDS-ecosystem-orange.svg?logo=nvidia)

</div>

## Content  
* [🧠 Overview](#overview)
* [🗂 Project Structure](#project-structure)
* [⚙️ Setup](#setup)
* [🚀 Usage](#usage)
* [📞 Contact and Support](#contact-and-support)  

## Overview  

In this project, we provide notebooks to compare the execution time of dataset operations using traditional **Pandas** (CPU) versus **NVIDIA’s cuDF**, a GPU-accelerated drop-in replacement for Pandas. This example is presented in two different formats:

- **Data Analysis Notebooks**: These notebooks use preprocessed datasets of varying sizes from **datafabric** folder in AI Studio. The evaluation is split across two sets of notebooks: one set using Pandas (CPU) and the other set using cuDF (GPU)—with performance metrics logged to **MLflow**.

## Project Structure  
```
├── docs                                                  # Directory containing UI screenshots of the logged results
├── notebooks                                             # Main notebooks for the project
│   ├── stock-analysis-with-pandas                        # Directory of notebooks using Pandas only (CPU)
|   |   └── run-workflow.ipynb                            # Notebook used for executing the pipeline using custom inputs and configurations
│   └── stock-analysis-with-pandas-and-cudf               # Directory of notebooks using cuDF (GPU)
|       └── run-workflow.ipynb                            # Notebook used for executing the pipeline using custom inputs and configurations
├── README.md                                             # Project documentation
└── requirements.txt                                      # Python dependencies (used with pip install)
```  

## Setup

### Step 0: Minimum Hardware Requirements

Ensure your environment meets the minimum compute requirements for smooth dashboard rendering and cuDF performance:

- **RAM**: 16 GB  
- **VRAM**: 4 GB  
- **GPU**: NVIDIA GPU

### Step 1: Create an AI Studio Project

- Create a new project in [Z by HP AI Studio](https://zdocs.datascience.hp.com/docs/aistudio/overview).

### Step 2: Set Up a Workspace

- Choose **RAPIDS Base** or **RAPIDS Notebooks** as the base image.

### Step 3: Clone the Repository

```bash
https://github.com/HPInc/AI-Blueprints.git
```

- Ensure all files are available after workspace creation.

---  

### Step 4: Add the Dataset to Workspace
1.  Download the **USA_Stocks** dataset from AWS S3 using the Datasets tab in your AI Studio project:
  - **Dataset Name**: `USA_Stocks`
  - **Dataset Source**: `AWS S3`
  - **S3 URI**: `s3://dsp-demo-bucket/rapids-data`
  - **Bucket Region**: `us-west-2`
2. Make sure that the dataset is in the `datafabric` folder inside your workspace.

### Step 5: Use a Custom Kernel for Notebooks  
1. In Jupyter notebooks, select the **aistudio kernel** to ensure compatibility.


## Usage 

### Step 1: Run the Notebooks

You can choose to run the **two data analysis 'run-workflow' notebooks** located in their respective directories inside the  `notebooks` folder to compare the performance of **vanilla Pandas** (CPU) and **cuDF** (GPU).  

For the two data analysis *run-workflow* notebooks, results are available both **within the notebook** and through **MLflow tracking**.

---

## Contact and Support  

- Issues: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting. 

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
