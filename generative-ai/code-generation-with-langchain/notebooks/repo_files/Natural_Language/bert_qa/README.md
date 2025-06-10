# Question and Answer with BERT
![alt](../../images/BERT_QA_image.png)

 [1. Project Setup on AI Studio](#1-project-setup-on-ai-studio)

 [2. How to use the experiment](#2-how-to-use-the-experiment)
 
 ---

 ## 1. Project Setup on AI Studio
 ---
 ### Workspace
 For this experiment, **we highly recommend yo to create a custom workspace on AI Studio using the Deep Learning GPU-based image**. 

 For extra libraries and specific versions, you can download our [requirements.txt](requirements.txt) and direclty upload it on the Add Python Libraries field.
 
 For the memory configurations, we recommend 16 GB for Memory and 7 GB for VRAM
 
 ![alt](../../images/BERT_QA_WORKSPACE_CONFIG.png)
---
 ### Accessing Jupyter Notebooks


Go to [Catalogue repository](https://github.azc.ext.hp.com/phoenix/ds-experiments/tree/draft/demo_catalogue) and copy the HTTPS URL. Next, go to your Project > Setup & Documentation > GitHub Repository and hit the CLONE GIT REPOSITORY button. Paste the URL, choose a local folder for the repository to be located and add it to the project.

![alt](../../images/CLONE_GITHUB_REPO.png)
 ---

 ### How to access the trained model
 ---
 #### Hugging Face

 ```python
 # Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("question-answering", model="morgana-rodrigues/bert_qa")
 ```
```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("morgana-rodrigues/bert_qa")
model = AutoModelForQuestionAnswering.from_pretrained("morgana-rodrigues/bert_qa")
```
---
#### Download from SharePoint (Private Group - AIS Team only)

##### [BERT_QA on SharePoint](https://hp.sharepoint.com/:f:/t/HPDataSciencePlatform/Eni6WF_x3mdOm8N3HOxooAcBmm3WSa18FnU5v8MCPE2Zjw?e=eljGpo)
---

---

 ## 2. How to use the experiment
---
Since we provide you the trained model and the mlruns from the training part of the experiment, there's no need for you to run it again. You can just show the Training.ipynb and the MLFlow on the Monitor tab for showing the results.

For deploying the model, you have to run the Deployment.ipynb notebook for registering the model using MLFlow. Then, you go to Published Servicies, create a new deploy, select the model and the (latest) version and start it.

![alt](../../images/BERT_QA_PUBLISHED_SERVICES.png)

After a while, you can click on the URL and access the Swagger UI for sending a context and a question about this context to the trained model. Click on POST/invocations > Try it out and edit the request body with a simple text for being your 'context' string key and a question about this context in the 'question' string key as the image shows below. Then, press 'Execute' and check the model response on the 'Server response'

![alt](../../images/BERT_REQUEST_BODY.png)
![alt](../../images/bert_body_response.png)

---