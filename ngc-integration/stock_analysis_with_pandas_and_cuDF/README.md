# 📈 Stock Analysis with Pandas and cuDF  

## Content  
- Overview  
- Project Structure  
- Setup  
- Usage  
- Contact and Support  

## Overview  

In this project, we provide notebooks to compare the execution time of dataset operations using traditional **Pandas** (CPU) versus **NVIDIA’s cuDF**, a GPU-accelerated drop-in replacement for Pandas. This example is presented in two different formats:

- **Data Analysis Notebooks**: These notebooks use preprocessed datasets of varying sizes from **datafabric** folder in AI Studio. The evaluation is split across two notebooks—one using Pandas (CPU) and the other using cuDF (GPU)—with performance metrics logged to **MLflow**.

## Project Structure  
```
├── README.md
├── notebooks
│   ├── stock_analysis_with_pandas.ipynb
│   └── stock_analysis_with_pandas_and_cuDF.ipynb
└── requirements.txt
```  

## Setup  

### Step 1: Create an AI Studio Project  
1. Create a **New Project** in AI Studio.   

### Step 2: Create a Workspace  
1. Select **RAPIDS Notebooks** as the base image.    

### Step 3: Verify Project Files  
1. Clone the GitHub repository:  
   ```
   git clone https://github.com/HPInc/aistudio-samples.git
   ```  
2. Navigate to `ngc-integration/stock_analysis_with_pandas_and_cuDF` to ensure all files are cloned correctly after workspace creation.  

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

You can choose to run the **two data analysis notebooks** located in the `notebooks` folder to compare the performance of **vanilla Pandas** (CPU) and **cuDF** (GPU).  

For the two data analysis notebooks, results are available both **within the notebook** and through **MLflow tracking**.

## Contact and Support  
- If you encounter issues, report them via GitHub by opening a new issue.  
- Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting.  

---

> Built with ❤️ using [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).
