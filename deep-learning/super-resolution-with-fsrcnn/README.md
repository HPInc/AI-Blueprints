# 🖼️ Image Super Resolution with FSRCNN

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-supported-orange.svg?logo=jupyter)
![TensorFlow](https://img.shields.io/badge/TensorFlow-used-ff6f00.svg?logo=tensorflow)
![FSRCNN](https://img.shields.io/badge/FSRCNN-model-blue.svg)
![Streamlit UI](https://img.shields.io/badge/User%20Interface-Streamlit-ff4b4b.svg?logo=streamlit)

</div>

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
│   └── streamlit-ui-image-super-resolution-with-fsrcnn.pdf             # UI screenshot
│   └── streamlit-ui-image-super-resolution-with-fsrcnn.png             # UI screenshot
│   └── swagger_UI_image_super_resolution_with_fsrcnn.pdf               # Swagger screenshot
│   └── swagger_UI_image_super_resolution_with_fsrcnn.png               # Swagger screenshot 
├── demo
│   └── streamlit-webapp/                                               # Streamlit UI
│   │  └── assets/                                                      # Logo assets
├── notebooks
│   └── register-model.ipynb                                            # Notebook for registering trained models to MLflow
│   └── run-workflow.ipynb                                              # Notebook for executing the pipeline using custom inputs and configurations  
│
├── README.md                                                           # Project documentation
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

- Download the `DIV2K dataset`

  - **Asset Name**: `DIV2K` 
  - **Source**: `AWS S3`
  - **S3 URI**: `s3://dsp-demo-bucket/div2k-data`
  - **Resource Type**: `public`
  - **Bucket Region**: `us-west-2`

- Make sure that the dataset is in the datafabric folder inside your workspace. If the dataset does not appear after downloading, please restart your workspace.

### 4 ▪ Clone the Repositoryy

```bash
https://github.com/HPInc/AI-Blueprints.git
```

- Ensure all files are available after workspace creation.

---

## Usage

### 1 ▪ Run the Notebook

Run the following notebook `run-workflow.ipynb`:
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

### 2 ▪ Run the Notebook

Run the following notebook `register-model.ipynb`:
- Log Model to MLflow
- Fetch the Latest Model Version from MLflow
- Load the Model and Run Inference

### 3 ▪ Deploy the Image Super Resolution with FSRCNN Service

- Go to **Deployments > New Service** in AI Studio.
- Name the service and select the registered model of register-model notebook.
- Choose a model version with **GPU**.
- Choose the workspace.
- Start the deployment.
- Note: This is a local deployment running on your machine. As a result, if API processing takes more than a few minutes, it may return a timeout error. If you need to work with inputs that require longer processing times, we recommend using the provided notebook in the project files instead of accessing the API via Swagger or the web app UI.

### 4 ▪ Swagger / raw API

Once deployed, access the **Swagger UI** via the Service URL.


Paste a payload like:

```
{
  "inputs": {
    "image": [
      "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAAcABwBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APAACzBVBJJwAO9dnp/wm8damu6Dw5dRjGf9IKw/+hkVPffCnWNJa7XVNV0Kxa1hErrNe/M2cnYqgElsAHpjkc1wlAODkV694W8c654t8M6n4TuvEctrrFw0cun3c0/lq+3AMJcDK5AyOeTkd+fPvGFn4gsvEtzF4m89tUG1ZJJjuMgUBVYN/EMKOe9YVXtK0bUtdvVs9LsZ7y4YgbIULYycZPoPc8V6lpfwh0/w7p66z8RdXj0y2z8llC4aWQ+mRn8lz9RXPfE3x1pvi46TYaPZTQadpMJghluWDSyrhQM9SMBe5Oc5NcBV7Tda1XRZJJNK1O8sXkG12tZ2iLD0JUjNQ3l9eahN517dT3MvTfNIXb16n6mq9Ff/2Q=="
    ]
  },
  "params": {}
}
```
And as response:
```
{
  "predictions": [
    "iVBORw0KGgoAAAANSUhEUgAAAHAAAABwCAIAAABJgmMcAABcKklEQVR4nK39V5Ntx5mgaX6uxZJbhDgCIEAyVWVWd9n09P+/nBkb66rqrupUVCBwzgm1xdKu3eciDkCQSWRmlc3da9sswixunu1ruX8eaNe3ozOK70UnxPLt00KU3LNWiOnDacNS7Xin2PjhsiHxuT9eDBKq552m48frj3sDoXe803T4eDWfm1w/Dq/da3L5ONjXzxW+fhoNCN3zXuPLp9F+bnT+NLnXVuj0MHkQuuedgvPD7EGojvcVnD/NHqTqWF/B5dPsPje6fJocSN2xrkLXT5P9l13j8eO4fW4yfRz+pFXPuprOH64rSN2ztqbLh+sKQveiqen24bogrnvZ1Nx9d5kxVb2ua758uy5qIG8AO9TtKsFL4ZRb7EhB0QQhpEWBZ7AmKC1NSSrDan2lxZaLLrBaX1WvXVYbfu...."
  ]
}
```

### 5 ▪ Launch the Streamlit UI

1. To launch the Streamlit UI, follow the instructions in the README file located in the `demo/streamlit-webapp` folder.

2. Navigate to the shown URL and view the Image Super Resolution.

### Successful UI demo

- Streamlit
![Handwritten Digit Classification Streamlit UI](docs/streamlit-ui-image-super-resolution-with-fsrcnn.png)

---

## Contact and Support  

- Issues: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting. 

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
