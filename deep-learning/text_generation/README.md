# Shakespeare text generation with RNN

### Content
- Overview
- Project Structure
- Setup
- Usage
- Contact and support

## Overview
The objective of this template is to show how to create a simple text generation with trained models from Hugging Face, one character per time using a dataset of Shakespeare's texts.

 ---

 ## Project Structure
```
├── code/                                        # Demo code
│
├── demo/                                        # Compiled Interface Folder
│
├── notebooks
    └── models/
│        └── decoder.pt                               # Reconstructs the input data from this compressed form to make it as similar as possible to the original input.
│        └── dict_torch_rnn_model.pt                  # Trained model for RNN_for_text_generation_Torch.ipynb
│        └── encoder.pt                               # Compresses the input into a compact representation and detain the most relevant features.
│        └── tf_rnn_model.h5                          # Trained model for the RNN_for_text_generation_TF.ipynb
│   ├── Deployment.ipynb                         # Notebook for registering the model using MLFlow
│   ├── RNN_for_text_generation_TF.ipynb         # Notebook for the TensorFlow trained model
│   ├── RNN_for_text_generation_Torch.ipynb      # Notebook for the Torch trained model
│   ├── deploy.py                                # Code to deploy        
├── README.md                                    # Project documentation
│                         
├── shakespeare.txt                              # Text from Shakespeare's Sonnet 1 that its gonna be used in this template
                                    
```
## Setup

 For the memory requirements, it's **recommended 40 GB for Memory and 4 GB for VRAM**

### Step 1: Create an AI Studio Project  
1. Create a **New Project** in AI Studio.   
2. (Optional) Add a description and relevant tags. 

### Step 2: Create a Workspace  
1. Select **Deep Learning with GPU** as the base image.

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
1. Obtain Text Data from the shakespeare.txt.
2. Prepare the textual data. It's needed to encode the data to provide the model a proper numerical representation of the text.
3. Create Training Batches for divide the dataset into smaller, manageable groups of data points that are fed into a machine learning model during the training process.
4. Create the GRU Model.
5. Train the model.
- Train the model with the selected epochs.
  
6. Generate the Predictions with the words 'Confidence' and 'Love'. The words can be changed.

### Optional:
Run the following notebook `/RNN_for_text_generation_Torch.ipynb`:
1. Obtain Text Data from the shakespeare.txt.
2. Prepare the textual data. It's needed to decode and encode the data to give the model a proper numerical representation of the text.
3. One Hot Encoding to convert categorical data into a fixed-size vector of numerical values.
4. Create Training Batches for divide the dataset into smaller, manageable groups of data points that are fed into a machine learning model during the training process.
5. Create the LSTM Model with the decoder and encoder files
6. Train the Network to do the Predictions
7. Generate the Predictions with the words 'Confidence' and 'Love'. The words can be changed.

### Deploy:
Run the following notebook `/Deployment.ipynb` for registering the model using MLFlow:
1. Obtain Text Data from the shakespeare.txt.
2. Load the model.
3. Register the model on MlFlow.
4. Test registered model

---
 ## Contact and Support  
- If you encounter issues, report them via GitHub by opening a new issue.  
- Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting.

---

> Built with ❤️ using Z by HP AI Studio.