# 🌍 Vacation Recommendation Agent  

## Content  
* [🧠 Overview](#overview)
* [🗂 Project Structure](#project-structure)
* [⚙️ Setup](#setup)
* [🚀 Usage](#usage)
* [📞 Contact and Support](#contact-and-support)

# Overview  
The **Vacation Recommendation Agent** is an AI-powered system designed to provide personalized travel recommendations based on user queries. It utilizes the **NVIDIA NeMo Framework** and **BERT embeddings** to generate relevant suggestions tailored to user preferences.  

# Project Structure  
```
├── README.md                                            # Project documentation
├── data                                                 # Data assets used in the project
│   └── raw
│       └── corpus.csv
├── demo                                                 # UI-related files
│   └── index.html
├── docs
│   ├── architecture.md                                  # Model Details and API Endpoints
│   └── ui_vacation.png                                  # UI screenshot
├── notebooks                                            # Main notebooks for the projects
│   ├── 00_Word_Embeddings_Generation.ipynb
│   └── 01_Bert_Model_Registration.ipynb
└── requirements.txt                                     # Python dependencies (used with pip install)
```  

# Setup 

### Step 0: Minimum Hardware Requirements

Ensure your environment meets the minimum compute requirements for smooth dashboard rendering and cuDF performance:

- **RAM**: ≥ 64 GB system memory  
- **VRAM**: 32 GB VRAM  
- **GPU**: NVIDIA GPU
- **Disk**: ≥ 32 GB free
- **CUDA**: Compatible CUDA toolkit (11.8 or 12.x) installed on your system

### Step 1: Create an AI Studio Project

- Create a new project in [Z by HP AI Studio](https://zdocs.datascience.hp.com/docs/aistudio/overview).

### Step 2: Set Up a Workspace 
- Choose **NeMo Framework** as the base image.    

### Step 3: Clone the Repository

```bash
https://github.com/HPInc/AI-Blueprints.git
```

- Ensure all files are available after workspace creation.

### Step 4: Add Project Assets  
1. Add the **Bertlargeuncased** (not **BertLargeUncasedForNemo**) model from the model catalog in AI Studio to your workspace. Use the `datafabric` folder inside the workspace to work with this model.

### Step 5: Use a Custom Kernel for Notebooks  
1. In Jupyter notebooks, select the **aistudio kernel** to ensure compatibility.

# Usage 

### Step 1: Generate Embeddings  
Run the following notebook to generate word embeddings and save the tokenizer:  
- `00_Word_Embeddings_Generation.ipynb`.  

### Step 2: Deploy the Service  
1. Execute `01_Bert_Model_Registration.ipynb` to register the BERT model in MLflow and create the API logic.  
2. Navigate to **Deployments > New Service** in AI Studio.  
3. Name the service and select the registered model.  
4. Choose an available model version and configure it with **GPU acceleration**.  
5. Start the deployment.  
6. Once deployed, click on the **Service URL** to access the Swagger API page.  
7. At the top of the Swagger API page, follow the provided link to open the demo UI for interacting with the locally deployed BERT model.  
8. Enter a search query (e.g., *"Suggest a budget-friendly resort vacation."*).  
9. Click **Get Recommendations** to view the results.  

### Successful Demonstration of the User Interface  

![Vacation Recommendation Demo UI](docs/ui_vacation.png)  

# Contact and Support

- Issues & Bugs: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

