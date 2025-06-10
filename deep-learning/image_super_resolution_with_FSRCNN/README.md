# 🖼️ Image Super Resolution with FSRCNN

### 📚 Content

* [🧠 Overview](#overview)
* [🗂 Project Structure](#project-structure)
* [⚙️ Setup](#setup)
* [🚀 Usage](#usage)
* [📞 Contact and Support](#contact-and-support)

 ## Overview

In this template, our objective is to increase the resolution of images, that is, to increase the number of pixels, using the FSRCNN model, a convolutional neural network model that offers faster runtime, which receives a low-resolution image and returns a higher-resolution image that is X times larger.

 ---
 ## Project Structure

 ```
├── docs/      
├── demo
│   └── streamlit-webapp/                                     # Streamlit UI
├── notebooks
│   ├── image_super_resolution_with_FSRCNN.ipynb               # Main notebook for the project
│
├── README.md                                                  # Project documentation
```

 ## Setup

### 0 ▪ Minimum Hardware Requirements

Ensure your environment meets the minimum compute requirements for smooth image classification performance:

- **RAM**: 16 GB  
- **VRAM**: 4 GB  
- **GPU**: NVIDIA GPU

### 1 ▪ Create an AI Studio Project 
1. Create a **New Project** in AI Studio.   
2. (Optional) Add a description and relevant tags. 

### Step 2: Create a Workspace  

- Choose **Deep Learning** as the base image.

### 3 ▪ Download the Dataset
1. This experiment requires the **DIV2K dataset** to run.
2. Download the dataset from `s3://dsp-demo-bucket/div2k-data` into an asset called DIV2K and ensure that the AWS region is set to ```us-west-2```.

### 4 ▪ Clone the Repositoryy

```bash
https://github.com/HPInc/AI-Blueprints.git
```

- Ensure all files are available after workspace creation.

---

## Usage

### 1 ▪ Run the Notebook
Run the following notebook `FSRCNN_DIV2K_AISTUDIO.ipynb`:
1. Model:
- Run the model architecture, which will do the feature extraction, shrinking, non-linear mapping, expanding and deconvolution.
2. Dataloader / preprocessing:
- The preprocessing of the DIV2K dataset will be done here.
3. Training and Validation:
- Train your FSRCNN model.
- Monitor metrics using the **Monitor tab**, MLflow, and TensorBoard.
4. Inference:
- Save the model and perform inference on the predicted image and the high-resolution image.
5. HR and LR image comparison:
- Compare the low-resolution and high-resolution images after training.

### 2 ▪ Local deployment on AI Studio

The local deployment should be done through the Deployments tab in AIStudio. Simply select the previously trained model, and then you will be able to perform super-resolution inferences on new images.

---

## Contact and Support

- Issues & Bugs: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**Z by HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).