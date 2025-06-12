# 📜💬 Shakespeare text generation with RNN

### 📚 Content

* [🧠 Overview](#overview)
* [🗂 Project Structure](#project-structure)
* [⚙️ Setup](#setup)
* [🚀 Usage](#usage)
* [📞 Contact and Support](#contact-and-support)

 ## Overview

The objective of this template is to show how to create a simple text generation with trained models from Hugging Face, one character per time using a dataset of Shakespeare's texts.

 ---

 ## Project Structure
```
├── code/                                             # Demo code
│                                   
├── data/                                             # Data assets used in the project
│    └── shakespeare.txt                              # Text from Shakespeare's Sonnet 1 that its gonna be used in this template
│
├── demo/                                             # Compiled Interface Folder
│
├── notebooks
    └── models/
│        └── decoder.pt                               # Reconstructs the input data from this compressed form to make it as similar as possible to the original input.
│        └── dict_torch_rnn_model.pt                  # Trained model for text_generation_with_RNN__Torch.ipynb
│        └── encoder.pt                               # Compresses the input into a compact representation and detain the most relevant features.
│        └── tf_rnn_model.h5                          # Trained model for the text_generation_with_RNN_TF.ipynb
│   ├── Deployment.ipynb                              # Notebook for registering the model using MLFlow
│   ├── text_generation_with_RNN_TF.ipynb             # Notebook for the TensorFlow trained model
│   ├── text_generation_with_RNN_Torch.ipynb          # Notebook for the Torch trained model
│   ├── deploy.py                                     # Code to deploy        
├── README.md                                         # Project documentation
                                    
```
## Setup

### 0 ▪ Minimum Hardware Requirements

Ensure your environment meets the minimum compute requirements for smooth performance:

- **RAM**: 16 GB  
- **VRAM**: 4 GB  
- **GPU**: NVIDIA GPU

### 1 ▪ Create an AI Studio Project

- Create a new project in [Z by HP AI Studio](https://zdocs.datascience.hp.com/docs/aistudio/overview).

### 2 ▪ Set Up a Workspace

- Choose **Deep Learning** as the base image.

### 3 ▪ Clone the Repository

```bash
https://github.com/HPInc/AI-Blueprints.git
```

- Ensure all files are available after workspace creation.

---

## Usage

### 1.1 ▪ Run the Notebook (Optional):
Run the following notebook `/RNN_for_text_generation_TF.ipynb`:
1. Obtain Text Data from the shakespeare.txt.
2. Prepare the textual data. It's needed to encode the data to provide the model a proper numerical representation of the text.
3. Create Training Batches for divide the dataset into smaller, manageable groups of data points that are fed into a machine learning model during the training process.
4. Create the GRU Model.
5. Train the model.
- Train the model with the selected epochs.
  
6. Generate the Predictions with the words 'Confidence' and 'Love'. The words can be changed.

### 1.2 ▪ Run the Notebook (Optional):
Run the following notebook `/RNN_for_text_generation_Torch.ipynb`:
1. Obtain Text Data from the shakespeare.txt.
2. Prepare the textual data. It's needed to decode and encode the data to give the model a proper numerical representation of the text.
3. One Hot Encoding to convert categorical data into a fixed-size vector of numerical values.
4. Create Training Batches for divide the dataset into smaller, manageable groups of data points that are fed into a machine learning model during the training process.
5. Create the LSTM Model with the decoder and encoder files
6. Train the Network to do the Predictions
7. Generate the Predictions with the words 'Confidence' and 'Love'. The words can be changed.
8. Integrate MLflow 

### Step 2: Deploy the Service  
1. Execute `/RNN_for_text_generation_Torch.ipynb` to register the model in MLflow and create the API logic.  
2. Navigate to **Deployments > New Service** in AI Studio.  
3. Name the service and select the registered model.  
4. Choose an available model version and configure it with **GPU acceleration**.  
5. Start the deployment.  
6. Once deployed, click on the **Service URL** to access the Swagger API page.  
7. At the top of the Swagger API page, follow the provided link to open the demo UI for interacting with the locally deployed model.  

---

## Contact and Support  

- Issues: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting. 

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).