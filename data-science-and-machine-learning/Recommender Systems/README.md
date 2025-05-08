# 🎥 Recommender Systems

## 📚 Contents

- Overview  
- Project Structure  
- Setup  
- Usage  
- Contact & Support

---

## 🧠 Overview

This Project shows how to do a movie recommendation system using Tensorflow.

---

## 🗂 Project Structure

```
├── model_artifacts/
│   └── movie_titles.csv
│
├── notebooks
│   └── Recommender Systems.ipynb             # Main notebook for the project              
├── README.md                                 # Project documentation
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
notebooks/Recommender Systems.ipynb
```

This will:

- Load and prepare the data
- Create the model architecture  
- Train the model
- Make inference  


---

## 📞 Contact & Support

- 💬 For issues or questions, please [open a GitHub issue](https://github.com/HPInc/aistudio-samples/issues).
- 📘 Refer to the official [AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview) for detailed instructions and troubleshooting tips.

---

> Built with ❤️ using [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).