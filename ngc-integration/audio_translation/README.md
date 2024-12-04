# Audio translation using NeMo models
<!-- ![alt](../../../images/BERT_QA_image.png) -->
<!-- colocar uma imagem aqui -->

 [1. Project Setup on AI Studio](#1-project-setup-on-ai-studio)

 [2. How to use the experiment](#2-how-to-use-the-experiment)
 
 ---

 ## 1. Project Setup on AI Studio
 ---
 ### Workspace
 For this experiment, **we highly recommend yo to create a workspace on AI Studio using the NeMo image**. 
 Also, the experiments will not run properly with less than 8GB of dedicated GPU memory
 
 <!-- ![alt](../../../images/BERT_QA_WORKSPACE_CONFIG.png) -->
 <!-- colocar uma imagem aqui ilustrando a criação do workspace customizado -->
---
 ### Accessing Jupyter Notebooks

Go to [Catalogue repository](https://github.azc.ext.hp.com/phoenix/ds-experiments/) and copy the HTTPS URL. Next, go to your Project > Setup & Documentation > GitHub Repository and hit the CLONE GIT REPOSITORY button. Paste the URL, choose a local folder for the repository to be located and add it to the project. You can also download it directly from the GitHub site

![alt](../../../images/CLONE_GITHUB_REPO.png)

 ### Setting up models
 Three different models from NGC are necessary to be downloaded for this example. To add these models in your project, you can do it when setting up the project - using the add assets option - or you can go through the Setup & Documentation tab. When clicking add assets, you should select NGC models than search for the following models on NGC repository. To make it easy, add the asset with the given name, so the destination folder is properly linked in the notebook.
  * First model performs Speech to Text, to be applied to english audio and convert to English text: stt_en_citrinet_1024_gamma_0_25-1.0.0 (to be added as an asset named STT-Citrinet)
  * Second model translate the text in English into Spanish: nmt_en_es_transformer12x2-1.0.0rc1 (to be added as an asset named en-es-transformer)
  * Third model perfors text to speech, generating an audio from the text in Spanish: tts_es_multispeaker_fastpitchhifigan-1.15.0 (to be added as an assed named FastPitch-HiFiGAN)

 ---

 ## 2. How to use the experiment
 ---
     This experiment consists of the notebook [english_to_spanish.ipynb ](english_to_spanish.ipynb). In the experiment, third cell loads the models from the assets downloaded from NGC - if the assets were named differently of the suggestion, you should fix it here. Also, this cell might take some time to run and, if you have 4GB or less of GPU memory available, an error might appear.
	 
	 Following cells will show an example, where a local audio in English is converted to text, than translated into a text in Spanish, to be then converted back into Audio (in Spanish). The last 2 cells will use MLFlow to log and register the composite execution as a new Model, to be used as a model service.
 ---

 ## 3. Local deployment on AI Studio
   fill in later

 ---