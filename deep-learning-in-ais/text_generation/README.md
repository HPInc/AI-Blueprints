# Shakespeare text generation with RNN

### Content
- Overview
- Project Structure
- Setup
- Usage
- Contact and support

## Overview
In this template we will show how to create a simple text generation with trained models from Hugging Face, one character per time using a dataset of Shakespeare's texts.

 ---

 ## Project Structure
```
├── code/                                        # Demo code
│
├── demo/                                        # Compiled Interface Folder
│ 
├── models/
│   ├── decoder.pt                               # Reconstructs the input data from this compressed form to make it as similar as possible to the original input.
│   ├── dict_torch_rnn_model.pt                  # Trained model for RNN_for_text_generation_Torch.ipynb
│   ├── encoder.pt                               # Compresses the input into a compact representation and detain the most relevant features.
│   ├── tf_rnn_model.h5                          # Trained model for the RNN_for_text_generation_TF.ipynb
│
├── notebooks
│   ├── Deployment.ipynb                         # Notebook for registering the model using MLFlow
│   ├── RNN_for_text_generation_TF.ipynb         # Notebook for the TensorFlow trained model
│   ├── RNN_for_text_generation_Torch.ipynb      # Notebook for the Torch trained model
│   ├── deploy.py                                # Code to deploy        
├── README.md                                    # Project documentation
│                         
├── shakespeare.txt                              # Text from Shakespeare's Sonnet 1 that its gonna be used in this template
                                    
```
## Setup

 For the memory requirements, we suggest to have **we recommend 40 GB for Memory and 4 GB for VRAM**

 We highly recommend you to create a custom workspace on AI Studio using GPU-based image.

### Step 1: Create an AI Studio Project  
1. Create a **New Project** in AI Studio.   
2. (Optional) Add a description and relevant tags. 

### Step 2: Create a Workspace  
1. Select **Deep Learning** as the base image.

### Step 3: Verify Project Files 
1. Clone the GitHub repository:  
   ```
   git clone https://github.com/HPInc/aistudio-samples.git
   ```  
2. Make sure the folder `deep-learning-in-ais/text_generation` is present inside your workspace.

---

## Usage

### Optional:
Run the following notebook `/RNN_for_text_generation_TF.ipynb`:
1. Get Text Data from the shakespeare.txt.
2. Prepare the textual data. We need to encode our data to give the model a proper numerical representation of our text.
3. Create Training Batches for divide the dataset into smaller, manageable groups of data points that are fed into a machine learning model during the training process.
4. Create the GRU Model.
5. Train the model.
- Train the model with the selected epochs
6. Generate the Predictions with the words 'Confidence' and 'Love'. You can change the words.

### Optional:
Run the following notebook `/RNN_for_text_generation_Torch.ipynb`:
1. Get Text Data from the shakespeare.txt.
2. Prepare the textual data. We need to encode our data to give the model a proper numerical representation of our text.
3. One Hot Encoding to convert categorical data into a fixed-size vector of numerical values.
4. Create Training Batches for divide the dataset into smaller, manageable groups of data points that are fed into a machine learning model during the training process.
5. Creating the LSTM Model with the decoder and encoder files
6. Train the Network to do the Predictions
7. Generate the Predictions with the words 'Confidence' and 'Love'. You can change the words.

### Deploy:
Run the following notebook `/Deployment.ipynb` for registering the model using MLFlow:
1. Go to Published Servicies.
2. Create a new deploy.
3. Select the model and the (latest) version and start it.
4. Click on the URL and access the Swagger UI for sending a initial word for the model and the size of the text.
5. Click on POST/invocations > Try it out and edit the request body.

---

#### Download from SharePoint (Private Group - AIS Team only)

##### [RNN models on SharePoint](https://hp.sharepoint.com/:f:/r/teams/HPDataSciencePlatform/Shared%20Documents/QA/%5BiUAT%5D%20Data%20sources/Models/RNNs?csf=1&web=1&e=1hgHAx)

---
 ## Contact and Support  
- If you encounter issues, report them via GitHub by opening a new issue.  
- Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting.

---

> Built with ❤️ using Z by HP AI Studio.