# ✍️ Handwritten digit classification with keras

## 📚 Contents

- Overview  
- Project Structure  
- Setup  
- Usage  
- Contact & Support

---

## 🧠 Overview

This project shows how to do a image classification, specifically digits of handwritten images, using TensorFlow and MNIST(Modified National Institute of Standards and Technology) dataset of handwritten digits. The MNIST dataset consists of a collection of handwritten digits from 0 to 9. 

---

## 🗂 Project Structure

```
├── demo
│   └── streamlit-webapp/                                             # Streamlit UI
├── notebooks
│   └── handwritten_digit_classification_with_keras.ipynb             # Main notebook for the project  
├── README.md                                                         # Project documentation
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

### Step 3: Clone the Repository

```bash
https://github.com/HPInc/aistudio-samples.git
```

- Ensure all files are available after workspace creation.

---

## 🚀 Usage

### Step 1: Run the Notebook

Execute the notebook inside the `notebooks` folder:

```bash
notebooks/handwritten_digit_classification_with_keras.ipynb
```

This will:

- Load and preprocess the MNIST data 
- Create the model architecture  
- Train the model
- Make inference
- Integrate MLflow 

### Step 2 ▪ Launch the Streamlit UI

1. To launch the Streamlit UI, follow the instructions in the README file located in the `demo/streamlit-webapp` folder.

2. Navigate to the shown URL and view the handwritten classification.

---

## 📞 Contact & Support

- 💬 For issues or questions, please [open a GitHub issue](https://github.com/HPInc/aistudio-samples/issues).
- 📘 Refer to the official [AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview) for detailed instructions and troubleshooting tips.

---

> Built with ❤️ using [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).