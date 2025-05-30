# ❓ Question and Answer with BERT

### 📚 Content

- Overview
- Project Structure
- Setup
- Usage
- Contact and support

 ## 🧠 Overview

 The Bidirectional Encoder Representations from Transformers (BERT) is based on a deep learning model in which every output is connected to every input, and the weightings between them are dynamically calculated based upon their connection. BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications.
  
 ---

## Project Structure
```
├── code/                                        # Demo code
│
├── demo/                                        # Compiled Interface Folder
│
├── notebooks
│   ├── Testing Mlflow Server.ipynb              # Notebook for testing the Mlflow server
│   ├── question_answering_with_BERT.ipynb       # Main notebook for the project
│   ├── deploy.py                                # Code to deploy                          
│
├── README.md                                    # Project documentation
│                                        
├── requirements.txt                             # Dependency file for installing required packages
                                    
```

## ⚙️ Setup

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

## 🚀 Usage

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

## 📞 Contact & Support

- 💬 For issues or questions, please [open a GitHub issue](https://github.com/HPInc/aistudio-samples/issues).
- 📘 Refer to the official [AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview) for detailed instructions and troubleshooting tips.

---

> Built with ❤️ using [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).