# 🎥 Recommender Systems with Tensorflow

## 📚 Contents

- Overview  
- Project Structure  
- Setup  
- Usage  
- Contact & Support

---

## 🧠 Overview

This project builds a simple **recommender system** for movies using **TensorFlow**.  
It trains on user-item interaction data to predict movie preferences with Model-based Collaborative Filtering.

---

## 🗂 Project Structure

```
├── docs/      
│   └── streamlit_ui_for_recommender_system.pdf               # UI screenshot
├── demo
│   └── streamlit-webapp/                                     # Streamlit UI
├── notebooks
│   └── recommender_systems_with_tensorflow.ipynb             # Main notebook for the project              
├── README.md                                                 # Project documentation
```

---

## ⚙️ Setup

### Step 0: Minimum Hardware Requirements

Ensure your environment meets the minimum compute requirements for smooth image classification performance:

- **RAM**: 16 GB  
- **VRAM**: 4 GB  
- **GPU**: NVIDIA GPU

### Step 1: Create an AI Studio Project

- Create a new project in [Z by HP AI Studio](https://zdocs.datascience.hp.com/docs/aistudio/overview).

### Step 2: Set Up a Workspace

- Choose **Deep Learning** as the base image.

### Step 3: Download the Dataset
1. This experiment requires the **tutorial_data dataset** to run.
2. Download the dataset from `s3://dsp-demo-bucket/tutorial_data/` into an asset called **tutorial** and ensure that the AWS region is set to ```us-west-2```.

### Step 4: Clone the Repository

```bash
https://github.com/HPInc/aistudio-samples.git
```

- Ensure all files are available after workspace creation.

---

## 🚀 Usage

### Step 1: Run the Notebook

Execute the notebook inside the `notebooks` folder:

```bash
notebooks/recommender_systems_with_tensorflow.ipynb
```

This will:

- Load and prepare the data
- Create the model architecture  
- Train the model
- Make inference
- Integrate MLflow  

### Step 2 ▪ Launch the Streamlit UI

1. To launch the Streamlit UI, follow the instructions in the README file located in the `demo/streamlit-webapp` folder.

2. Navigate to the shown URL and view the predicted movies recommendations.

---

## 📞 Contact & Support

- 💬 For issues or questions, please [open a GitHub issue](https://github.com/HPInc/aistudio-samples/issues).
- 📘 Refer to the official [AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview) for detailed instructions and troubleshooting tips.

---

> Built with ❤️ using [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).