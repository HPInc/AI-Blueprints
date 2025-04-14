# Question and Answer with BERT

### Content
- Overview
- Project Structure
- Setup
- Usage
- Contact and support

## Overview
 The Bidirectional Encoder Representations from Transformers (BERT) is based on a deep learning model in which every output is connected to every input, and the weightings between them are dynamically calculated based upon their connection. BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications.
  
 ---

## Project Structure
```
├── code/                                        # Demo code
│
├── demo/                                        # Compiled Interface Folder
│
├── notebooks
│   ├── Deployment.ipynb                         # Notebook for registering the model using MLFlow
│   ├── Testing Mlflow Server.ipynb              # Notebook for testing the Mlflow server
│   ├── Training.ipynb                           # Notebook for the trained model
│   ├── deploy.py                                # Code to deploy                          
│
├── README.md                                    # Project documentation
│                                        
├── requirements.txt                             # Dependency file for installing required packages
                                    
```

## Setup

For the memory configurations, it's **recommend 16 GB of RAM and 8 GB for VRAM** for training the model. For inference in GPU, 4 GB of VRAM are sufficient.

### Step 1: Create an AI Studio Project  
1. Create a **New Project** in AI Studio.   
2. (Optional) Add a description and relevant tags. 

### Step 2: Create a Workspace  
1. Select **Deep Learning** as the base image.
- Libraries in the requirements.txt must be available to run the experiment.

### Step 3: Verify Project Files 
1. Clone the GitHub repository:  
   ```
   git clone https://github.com/HPInc/aistudio-samples.git
   ```  
2. Make sure the folder `deep-learning-in-ais/bert_qa` is present inside your workspace.

---

## Usage
### Optional:
Run the following notebook `/Training.ipynb`:
1. Download the dataset from the HuggingFace datasets repository.
2. Tokenize, preparing the inputs for the model.
3. Load metrics and transforms the output model(Logits) to numbers.
4. Train, using the model:
```
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint_bbc)

```
5. Complete the training evaluation of the model.
6. Create a question-answering pipeline from transformers and pass the model to it.

### Deploy:
Run the following notebook `/Deployment.ipynb`(The same deployment can be achieved by running the deploy.py file):
1. In the Model section, load a Transformer model saved on Hugging Face to use it locally.
2. Register the model.
3. Test the model that was registred.

 ## Contact and Support  
- For issues, please report them by opening a new issue on GitHub.  
- Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting.

---

> Built with ❤️ using Z by HP AI Studio.