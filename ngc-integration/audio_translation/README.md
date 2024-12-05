# Audio translation using NeMo models
<!-- ![alt](../../../images/BERT_QA_image.png) -->
<!-- colocar uma imagem aqui -->

 [1. Project Setup on AI Studio](#1-project-setup-on-ai-studio)

 [2. How to use the experiment](#2-how-to-use-the-experiment)
 
 ---

 ## 1. Project Setup on AI Studio
 ---
 ### Workspace
 For this experiment, **create a workspace on AI Studio using the NeMo image available from NVIDIA's NGC Catalog**. 
 
 **NOTE:** experiments will not run properly with less than 8GB of dedicated GPU memory
 
 <!-- ![alt](../../../images/BERT_QA_WORKSPACE_CONFIG.png) -->
 <!-- colocar uma imagem aqui ilustrando a criação do workspace customizado -->
---
 ### Accessing Jupyter Notebooks

1. Go to [Catalogue repository](https://github.com/HPInc/aistudio-samples) and copy the HTTPS URL. 
2. Next, go to your Project > Project Setup > Setup > GitHub Repository and hit the CLONE GIT REPOSITORY button. 
3. Paste the URL as the path in your AI Studio project
4. Choose a local folder for the repository to be located and add it to the project. 

**NOTE:** You can also download it directly from the GitHub site and connect to it by adding a dataset in the project.


 ### Setting up models
 Three different models from NVIDIA's NGC Catalog are necessary to be downloaded for this example. To add these models in your project, you can do it when setting up the project - using the add models option in the second step of the project set-up - or you can go through the Assets tab to add models from the Catalog at any time. 
 
 When clicking add assets, you should select 'models' than search for the following models on NGC repository. After adding, proceed to set-up your workspace by selecting the NeMO Framework image available from the NVIDIA NGC Catalog and click 'Create Workspace.'
 
1. First model performs Speech to Text, to be applied to english audio and convert to English text:
  ```stt_en_citrinet_1024_gamma_0_25-1.0.0 ```

2. Second model translate the text in English into Spanish:
  ```nmt_en_es_transformer12x2-1.0.0rc1 ```

3. Third model perfors text to speech, generating an audio from the text in Spanish:
```tts_es_multispeaker_fastpitchhifigan-1.15.0```

  stt_en_citrinet_1024_gamma_0_25-1.0.0 will be added as an asset named **STT-Citrinet**. nmt_en_es_transformer12x2-1.0.0rc1 will be added as an asset named **en-es-transformer**. tts_es_multispeaker_fastpitchhifigan-1.15.0 will be added as an asset named **FastPitch-HiFiGAN**.

4. After selecting the NGC models, select and download the NGC NeMO image.
 
5. As the workspace container builds, navigate to Assets > Models to check that your models have downloaded. Download them, if they have not.
 
6. Open the workspace and confirm models are in the 'datafabric' folder in Jupyter.

 ---

 ## 2. How to use the experiment
 ---
   This experiment consists of the notebook [english_to_spanish.ipynb ](english_to_spanish.ipynb). In the experiment, third cell loads the models from the assets downloaded from NGC - if the assets were named differently of the suggestion, you should fix it here. Also, this cell might take some time to run and, if you have 4GB or less of GPU memory available, an error might appear.
	 
   Following cells will show an example, where a local audio in English is converted to text, than translated into a text in Spanish, to be then converted back into Audio (in Spanish). The last 2 cells will use MLFlow to log and register the composite execution as a new Model, to be used as a model service.

   Run code or use terminal to run deploy_ms.py script.
   
 ---

 ## 3. Local deployment on AI Studio
   Check that model has registered to MLFlow by navigating to the Monitor tab. With the model registered, navigate to Deployment and deploy the service. You can then click the play button to publish the service for local inference to a Swagger API service.

 ---
