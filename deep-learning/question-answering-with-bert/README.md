# ❓ Question and Answer with BERT

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-supported-orange.svg?logo=jupyter)
![HuggingFace](https://img.shields.io/badge/Hugging--Face-model-yellow.svg?logo=huggingface)
![BERT](https://img.shields.io/badge/BERT-used-lightgreen.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-used-ff6f00.svg?logo=tensorflow)
![React UI](https://img.shields.io/badge/User%20Interface-React-61DAFB.svg?logo=react)

</div>

### 📚 Content

* [🧠 Overview](#overview)
* [🗂 Project Structure](#project-structure)
* [⚙️ Setup](#setup)
* [🚀 Usage](#usage)
* [📞 Contact and Support](#contact-and-support)

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
│   ├── other-notebooks                          
│   │     └── deploy.py                          # Code to deploy 
│   │     └── Testing Mlflow Server.ipynb        # Notebook for testing the Mlflow server  
│   │
│   ├── register-model.ipynb                                          # Notebook for registering trained models to MLflow
│   ├── run-workflow.ipynb                                            # Notebook for executing the pipeline using custom inputs and configurations                         
│
├── README.md                                    # Project documentation
│                                        
├── requirements.txt                             # Dependency file for installing required packages
                                    
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

### 1 ▪ Run the Notebook
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
7. Integrate MLflow 

### 2 ▪ Deploy
1. Run the following notebook `/question_answering_with_BERT.ipynb`(The same deployment can be achieved by running the deploy.py file): 
2. Navigate to **Deployments > New Service** in AI Studio.  
3. Name the service and select the registered model.  
4. Choose an available model version and configure it with **GPU acceleration**.  
5. Start the deployment.  
6. Once deployed, click on the **Service URL** to access the Swagger API page.  
7. At the top of the Swagger API page, follow the provided link to open the demo UI for interacting with the locally deployed model.  


---

# Contact and Support  

- Issues: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting. 

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
