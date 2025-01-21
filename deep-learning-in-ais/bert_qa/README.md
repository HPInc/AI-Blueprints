# Question and Answer with BERT

 [1. Project Setup on AI Studio](#1-project-setup-on-ai-studio)

 [2. Structure of the experiment](#2-structure-of-the-experiment)
 
 ---

 ## 1. Project Setup on AI Studio

 For this experiment, **create a custom workspace on AI Studio using the Deep Learning GPU-based image**. Libraries in the requirements.txt must be available to run the experiment. We highly recommend setting up a custom workspace with the libraries present on the requirements.txt file.
 
 For the memory configurations, we recommend 16 GB of RAM and 8 GB for VRAM for training the model. For inference in GPU, 4 GB of VRAM are sufficient.
 
## 2. Structure of the experiment

   ### Running inference and deploying models

   Deployment.ipynb notebook has all the necessary code to download the available model from Hugging Face and running the model for inference inside the notebook. This notebook also contains the code to log and register the model in ML Flow, for deploying the model locally. 

   The same deployment can be achieved by running the deploy.py fila

   ### Retraining the model

   For those interested in retraining the model and understand deeply how to work with deep learning, the Training.ipynb notebook provides the code for that.