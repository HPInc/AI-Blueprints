# Super Resolution

### Content
- Overview
- Project Structure
- Setup
- Usage
- Contact and support

 ## Overview

In this template, our objective is to increase the resolution of images, that is, to increase the number of pixels, using the FSRCNN model, a convolutional neural network model that offers faster runtime, which receives a low-resolution image and returns a higher-resolution image that is X times larger.

 ---
 ## Project Structure

 ```
├── notebooks
│   ├── FSRCNN_DIV2K_AISTUDIO.ipynb               # Main notebook for running image super-resolution
│
├── README.md                                     # Project documentation
```
 ## Setup

 For memory requirements, we recommend **at least 4GB of RAM and 4GB of dedicated GPU RAM**.

 Similar configurations for a CPU-only experiment will work, but we do not recommend this due to the excessive training time.

### Step 1: Create an AI Studio Project  
1. Create a **New Project** in AI Studio.   
2. (Optional) Add a description and relevant tags. 

### Step 2: Create a Workspace  
1. Select **Deep Learning** as the base image.

### Step 3: Download the Dataset
1. This experiment requires the **DIV2K dataset** to run.
2. Download the dataset from `s3://dsp-demo-bucket/div2k-data` into an asset called DIV2K and ensure that the AWS region is set to ```us-west-2```.

### Step 4: Verify Project Files 
1. Clone the GitHub repository:  
   ```
   git clone https://github.com/HPInc/aistudio-samples.git
   ```  
2. Make sure the folder `deep-learning-in-ais/super_resolution` is present inside your workspace.

---

## Usage

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

 ## Local deployment on AI Studio

The local deployment should be done through the Deployments tab in AIStudio. Simply select the previously trained model, and then you will be able to perform super-resolution inferences on new images.

 ---

 ## Contact and Support  
- For issues, please report them by opening a new issue on GitHub.  
- Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting.

---

> Built with ❤️ using Z by HP AI Studio.
