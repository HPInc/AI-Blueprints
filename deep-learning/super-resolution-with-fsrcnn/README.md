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


### 3 ▪ Swagger / raw API

Once deployed, access the **Swagger UI** via the Service URL.


Paste a payload like:

```
{
  "inputs": {
    "image": [
      ""/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAAcABwBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APAACzBVBJJwAO9dnp/wm8damu6Dw5dRjGf9IKw/+hkVPffCnWNJa7XVNV0Kxa1hErrNe/M2cnYqgElsAHpjkc1wlAODkV694W8c654t8M6n4TuvEctrrFw0cun3c0/lq+3AMJcDK5AyOeTkd+fPvGFn4gsvEtzF4m89tUG1ZJJjuMgUBVYN/EMKOe9YVXtK0bUtdvVs9LsZ7y4YgbIULYycZPoPc8V6lpfwh0/w7p66z8RdXj0y2z8llC4aWQ+mRn8lz9RXPfE3x1pvi46TYaPZTQadpMJghluWDSyrhQM9SMBe5Oc5NcBV7Tda1XRZJJNK1O8sXkG12tZ2iLD0JUjNQ3l9eahN517dT3MvTfNIXb16n6mq9Ff/2Q==""
    ]
  },
  "params": {}
}
```

---

## Contact and Support  

- Issues: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting. 

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
