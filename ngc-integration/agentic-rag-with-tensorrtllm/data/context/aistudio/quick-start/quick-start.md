---
sidebar_label: 'AI Studio Quick Start Guide'
sidebar_position: 1
---

# AI Studio Quick Start Guide

Use this guide for a streamlined way to find answers and get started with AI Studio.

## What is AI Studio (AIS)?

AI Studio is an easy, simple, and safe way to create and deploy on-premises AI projects.

## Who is AI Studio For?

If you are an AI developer struggling with cloud computing limitations, you might be facing challenges like:

1. Hours spent reimaging computers or adapting shared models.

2. Difficulty organizing and documenting diverse data science tools and projects.

AI Studio tackles these issues head-on. It’s an enterprise-grade platform designed for seamless collaboration in local compute environments. By containerizing development environments and syncing them across users and machines, AIS simplifies project sharing and management. Built-in integrations with data hubs and model repositories, like NVIDIA's NGC catalog, speed up model development and deployment.

## Technical Requirements

#### Hardware:
- Windows 10 or 11 or Linux Ubuntu 22.04 LTS on a workstation

- GPU is not required, but is strongly recommended. AI Studio currently supports NVIDIA GPUs with 528.89 drivers or newer.

#### Software:
- Win OS requires Windows Subsystem for Linux (WSL). AIS installs WSL if not present.

- Git is required for github cloning, but not a requirement to use the app.
Git Repository Path - 
https://github.com/HPInc/aistudio-samples.git


- Internet access for web synching of project metadata.

## Activate, Install, and Configure AI Studio

- **Activate** (Admins Only): Click the link from the email you received.

    :::note
    Do not follow automated onboarding steps after clicking the activation link if you are an EAP user. Instead use the EAP installers emailed separately.
    :::

- **Download & Install**: Download the Windows or Linux installer provided to you and install AI Studio.

- **Sign in**: Log in with your existing HPID or create one if you don't already have one.

    :::note
    The first member of your team to do this will be granted admin access.
    :::

- **[Invite users](/docs/aistudio/account/team-settings.md)**: From the Account tile, navigate to Team Settings and select **Invite New Members** to invite your team.

- **[Create Project](/docs/aistudio/using-aistudio/projects/navigating.md)**: From the Projects page, click on **New Project**.

    - Add a description, tags, and decide if the project should be shared or private.
    
    - Point to a Github repo for code management (optional).
    
    - Point to data repositories (optional). You can add data anytime.

        :::note
        If a remote store, you will need access privileges.
        :::

    - Search for and get NVIDIA NGC GPU optimized pre-trained models (optional).
    
    - Define your workspace:
        
        - Use a base image.

            OR

        - Search for NVIDIA NGC GPU optimized containers and frameworks.
        
- **[Create a Custom Workspace](/docs/aistudio/using-aistudio/workspaces/custom-workspace.md)**: Before creating the project, you can add additional pip packages and libraries to the base images. To do this click **Add Custom Libraries** under the workspace name and either type the package name and version or upload a requirements.txt file.
    
- **[Configure Notebooks](/docs/aistudio/using-aistudio/workspaces/reusing-workspace.md)**: After you create your project the workspace will start, pull in your assets, and let you use them directly in the notebook.
    
- **[Metrics](/docs/aistudio/using-aistudio/monitoring.md)**: AIS has MLFlow (PyTorch) and Tensorboard (Tensorflow) built-in. Include hooks in your code to collect, track, and visualize model metrics. To view, navigate to the **Monitor** tab.
    
- **[Register a Model](/docs/aistudio/using-aistudio/model-deployment/register-model-scikit-learn.md)**: AIS uses MLFlow model registry for version control and to store models. To save a model to an MLFlow model registry, include MLFlow hooks in your code.
    
- **[Deploy](/docs/aistudio/using-aistudio/model-deployment/model-deployment-overview.md)**: AIS can locally deploy models registered to MLFlow using the Swagger API. Make sure you registered your model to MLFlow, then navigate to Published Services and deploy a model following the steps. Once published, click the play button to preview for local inference.
    
- **[Jobs](/docs/aistudio/using-aistudio/jobs/new-job.md)**: Once your model is complete, you can schedule local jobs to work at specific times and routines. Navigate to the ***Overview*** page in a project and click on **New Job.
    
Visit the [other guides on zDocs](/docs/aistudio/overview.md) for use cases, step-by-step guides, and additional information on AI Studio.

### (EAP Users Only) Activate, Download, and Install AI Studio

- **Activate**: Click on the link from the email you received (If you need to search for an email from aistudio@hp.com titled “AI Studio Account Invitation”).

    :::note
    Do not follow automated onboarding steps after clicking the activation link if you are an EAP user. Instead use the EAP installers emailed separately.
    :::

- **Download Installer**: Download the Windows or Linux installer provided to you and install AIS on your device.

- **Edit the configuration file (Required for v1.35.x)**: Go to `C:\Users\<yourusername>\AppData\Local\HP\AIStudio`in the file explorer (this may be a hidden folder and you will have to manually type in the path in the top bar) and open config.yaml in a text editor. Then add the text below to the end of the file (see screenshot below for an example) and save it.

    ngcconfig:
    nemoversionpin: "23.10"

    :::note
    This edit is required due to issues with the latest container image from NVIDIA. This will be fixed in a future release of AI Studio, but for now you need manually update it for these demos to work properly.
    :::

    ![Text Editor](/img/text_editor.png)

- **Open the AI Studio App and Sign in**: Log in with your HPID or create one, you will have admin access.

- **(Optional) Invite users**: From the Account tile, navigate to Team Settings and click **Invite New Members** to invite your team.



## English to Spanish Translation Example With NVIDIA NEMO

### Overview

This example should familiarize with all of the major concepts in AI Studio. It is intended for Software Engineers, Machine Learning Engineers, Data Scientists, or Data Analysts that currently write code either locally or in cloud services and are interested in developing locally with AI Studio.

### Expectations
You should expect to spend ~ 30 min – 1 hour following this guide to install and configure AI Studio and create a project using one of our example codebases and another 15-20 min bringing in one of your own repos from github and making small updates to get the most out
of AI Studio’s features.

### Technical Requirements

#### Hardware:
- A discrete GPU is not required to use AI Studio, but for this demo we are going to walk through a demo that **requires an NVIDIA GPU with at least 8 GB of VRAM**. AI Studio currently supports NVIDIA GPUs with 528.89 drivers or newer.

#### Software:
- Windows 10 or 11 or Linux Ubuntu 22.04 LTS

- Windows OS requires Windows Subsystem for Linux (WSL). AIS installs WSL if not present.

- Git is required for github cloning, but not a requirement to use the app.

- Internet access is required for downloading container images and for web syncing of project metadata.

### Creating Your First AI Studio Project
Projects are the main concept AI Studio is built around. They should be used to house all the information needed for someone to recreate a data science experiment, model training or data processing activity. To do this you will need to make a few changes to the way you may normally develop but the benefits are that when you are coming back to a project at a later date, handing off work or onboarding a new team member the setup time for that new person to continue from where you left off should be basically instant.

#### Create the Project

:::note
Before proceeding with this tutorial, ensure you have sufficient disk space, as the image download will require 28GB. This precaution helps avoid model download failures due to insufficient storage. If needed, refer to the guidelines on freeing up storage, including uninstallation or more technical methods - [Removing Expired Images | Z by HP AI & Data Science Solutions](https://zdocs.datascience.hp.com/docs/aistudio/using-aistudio/workspaces/removing-images).
:::

1. From the Projects page, click **New Project**.

2. Add Project Name: **Audio Translation using NeMo models**

    Add description: **Sample Project to demonstrate how to use a NeMo model inside an AI Studio Project**.

    :::tip

    NeMo™, which stands for "Neural Modules," by NVIDIA is an end-to-end platform used for developing custom generative AI,including large language models (LLMs), vision language models (VLMs), retrieval models, video models, and speech AI—anywhere. 
    This platform helps developers create AI applications more efficiently, leveraging NVIDIA's GPU acceleration to optimize performance and scalability.     
    Learn more about NVIDIA NeMo at [Build Custom Generative AI | NVIDIA NeMo](https://www.nvidia.com/en-us/ai-data-science/products/nemo/) 

    :::

3. Select between a Team Project (viewable and runnable by anyone in the
account) or Restricted Project (private to you and the account owners). For this demo (and in most cases outside of sensitive work) set it to ***Team***.

4. Point to Github repo for code management. The first time you do this you will be prompted to authenticate and link AI Studio with your github. Even though this is a public github repo, you will have to have a github account or create one if you don’t have one already. For this demo use:
https://github.com/HPInc/aistudio-samples.git

You will also need to select a local folder for the git repo to clone into. This is up to you but generally making an **aistudio** or a **repos** folder inside of your **Documents** folder is a good place to store code repositories. Simply create a new folder and point to it in AI Studio when creating your project.

5. Leave “Configure git for autocrlf” checked if you are on windows. 

6. Add tags: NGC, NeMo, and Audio.

:::tip

Adding tags will provide searchability as you build more projects. 

:::

7. Select Continue. 

8. Connect your data (instead of Add Assets) matches the UI. 

9. **Add Assets**:
    - You can point to existing connections to remote data, like S3 or Azure Blob storage using the Datasets tab (or add a new connection by clicking **New Asset**). You can add assets to a project anytime.

        :::note
        If a remote store, you will need to configure your access and
        have proper privileges.
        :::

    - For this Demo you will need 3 different NGC models. Search and add
    the following NVIDIA NGC GPU optimized pre-trained models to your
    project:

        - First model performs Speech to Text, to be applied to english
        audio and convert to English text:

            Name: **STT En Citrinet 1024 Gamma**

            Model: **stt_en_citrinet_1024_gamma_0_25.nemo**

        - Second model translate the text in English into Spanish:

            Name: **NMT En Es Transformer12x2**

            Model: **nmt_en_es_transformer12x2.nemo**

        - Third model performs text to speech, generating an audio from
        the text in Spanish:

            Name:
            **TTS Es Multispeaker FastPitch**

            Models:
            **tts_es_fastpitch_multispeaker.nemo**
            **tts_es_hifigan_ft_fastpitch_multispeaker.nemo**

10. **Define your workspace**:
To achieve the ability to run code consistently across different machines and different operating systems, code execution in AI Studio runs inside of container images. This allows for a consistent developer experience across different computers and operating systems, but comes with a few changes to your typical workflow which we will walk you through with our example project. In general you have 2 main options:

    - Use an AI Studio base image

        OR

    - Search for NVIDIA NGC GPU optimized containers and frameworks.
    
    For our demo select the ***NeMo Framework*** image from the NVIDIA NGC
    images. Click the ***download*** button (this is a one time download but since it is a larger image it will take a while so maybe go get yourself a coffee and check back in 20 min depending on your internet speed). Next give your Workspace a name, since this workspace is based off the NeMo Framework and we added no additional libraries you can just name it ***NeMo Framework***.

    #### Custom Workspaces:
    You also have the option when selecting a base image to add additional pip packages and libraries to the base images. To do this click ‘Add Custom Libraries’ under where you name the workspace and either type the package name and version or upload a requirements.txt file. You can click Validate after to test you have correctly added the additional python libraries. In practice you should give your customized workspaces descriptive names for the types of packages you have added.

### Running the Project

1. **Download Project Assets**:

    In the project setup, we added 3 models from the NGC catalog. You should be able to see them by going to the assets tab and selecting the models sub-tab.**Click the download icon** next to each model. Once the models are downloaded you can move on to start the workspace. 

    [Translation Assets](/img/translation_assets.png)

2. **Start the Workspace**:

    Now that you have created your project you can spin up the container image. To do this **click the play button** inside of the *Nemo Framework Workspace* card you created and you will be automatically brought into the Workspace tab. To turn off a workspace navigate to the *Overview* tab in a project and click the pause button inside of the workspace. If you accidentally close the workspace notebook tab you can open a new one by clicking on **Open Jupyter**, which looks like a terminal icon.
    
    ![Audio Translation](/img/audio_translation.png)

3. **Navigating the Workspace File structure**:

    The top level folder in a workspace always contains 3 folders (in your case it will have 4 folders since you cloned a git repo):

        - **aistudio-samples** will contain the git repository that you cloned in the
        project setup. For other projects this folder will be named the same as
        the git repository name.

        - **datafrabric** will contain any Assets you added from connections to s3,
        azure blob or other datasets or models in the Assets tab. In this case
        each of the 3 models will have their own folder.

        - **local** is a folder that mounts to your local machine and allows you to
        store files locally on your computer but are not shared with other users in your account. You will see that your github information is stored in a
        configuration folder there.
            :::tip
            The local folder can be a good place to store model files from
            providers AI Studio doesn’t natively support like HuggingFace or Ollama (which will usually require setting an environment variable to control where they download to ie: `HF_HOME = /home/jovyan/local` for hugging face models).
            :::

        - **shared** is a folder that allows for peer to peer data transfer between
        collaborators. Any files you add will be shared with any other
        collaborators in your account if you are both online at the same time.

            :::warning
            Since AI Studio’s workspaces are based on containers if you create or save any work outside of the above 4 folders when the container is shut down your file will be erased. To avoid losing data if you are working on project code it should be stored in the git repository folder, local or shared folders and data should be saved to the datafabric, local or shared folders.
            :::

4. **Running the Demo**:

    - Navigate to the demo notebook (aistudio-samples/ngcintegration/audio_translation/english_to_spanish.ipynb).

    - Run the notebook cells one by one (or click the double play button [or fast forward button however you call it] to run all cells).
    
        You will see the notebook loads in the 3 NeMo models, transcribes text from an audio sample, translates it from English to Spanish and creates a Spanish audio clip from the translated text. The last 2 cells have code that creates, logs and registers the model with MLflow.

5. **Reviewing the MLflow experiment tracking**:

    The Notebook logged and registered a model in MLflow to see that model click the Monitor tab then open the MLflow sub tab. You should see an experiment named Nemo_translation and a run named NeMo_en_es_translation, which contains MLflow logs of the model. If you are interested in adding MLflow logging to your code take a look at the following documentation from MLflow: [MLflow Tracking Quickstart](https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html).

6. **Deploy Demo Locally**:

    AIS can locally deploy models registered to MLFlow using Swagger API. To do so, you need a registered model in MLFlow, which was done when you ran the notebook code in the previous step. *Navigate to ***Deployments*** tab* and deploy a model locally by clicking on **New Service**.

        Use the following values:
        - Service Name: **Demo Deployment**
        - Select your model: **nemo_en_es**
        - Choose model version: **1**
        - GPU Configuration: **With GPU**
        - Choose Workspace: **Nemo Framework**

        ***Then Click Deploy***
        To launch the service **click the play icon** and wait a couple of minutes for the service to launch. When the service spins up it will launch a browser window with the swagger api site. To see the nicer front end built for this demo *click the black bar at the top of the swagger api site* that says “Click to see a demonstration of the service” to preview the demo which has a webapp built on top of the swagger API.

7. **Schedule a Batch Job**:

    Batch jobs can be scheduled as one time or recurring runs. They will run if the computer is on with AI Studio running. To do so navigate to the Overview page in a project and click **New Job**. As a simple test create a batch job with the following values:
        - Job Name: **Demo test job**
        - Date: **the current date**
        - Time: **~5 min from when you are creating it**
        - Workspace: **Nemo Framework**
        - Branch/Tag or Commit: **origin/main**
        - Script Path: **aistudio-samples/ngcintegration/audio_translation/deploy_ms.py**
        - Script Arguments: **[Leave this section blank]**

        Then, add ***all 3 model files*** using the check boxes, select your GPU and then click **Create Job**.

        Wait for the job to run (you will see the status change from *Active* to *Running* to *Finished*) and after it completes you will see Job Results populated and if you click the three dots to the right you can see the logs and any artifacts created.

Now that you have walked through an example of how to create a project in AI Studio it is your turn to make a Project with your own code! Make sure to find any places you are pip installing something in a notebook and add those to the custom workspace requirements and feel free to play around with the variety of NGC models and containers.

Visit the [other guides on zDocs](/docs/aistudio/overview.md) for use cases, step-by-step guides, and additional information on AI Studio. You can also check out the instructional videos on our [Youtube channel](https://www.youtube.com/playlist?list=PLgdwW86oDaa6v_3SOieCXihop57KoUHxw) or join AI compute discussions in the [community site](https://community.datascience.hp.com/)!


## BERT Q&A Demo Example

A BERT transformer encoder processes text by understanding context and meaning through attention mechanisms. It generates representations for words based on surrounding words. 
 
For this example, we'll use BERT to identify relationships in text, enabling it to match a question with the most relevant part of a given passage to provide you with an accurate answer. No additional data is required. 

Materials provided include code, the model, and a web app for local deployment. Once the model is deployed, you can interrogate the model for inference through the Swagger API interface or click on the web app link at the top of the Swagger page for a web app.

### Setup

1. From the Projects page, click **New Project**.

2. Name the project, then fill out description details (optional).

3. Use [this git repo link](https://github.com/HPInc/aistudio-samples) to clone the project.

4. Select a local destination to clone the repository to.

    :::tip
    Create a dedicated folder for repository clones if you don’t already have one. We recommend creating a ‘data’ folder on your desktop.
    :::

5. Click Continue

    :::note
    You can skip adding datasets and models in this step by clicking **Continue** again.

6. Select the Deep Learning GPU image (10.9 GB).

7. Name the workspace.

8. If you have not downloaded this image before, click on **Download Image**.

9. Click **Add Custom Libraries**, add your custom libraries, then click the button to upload **Requirements.txt**.

10. Navigate to where you cloned the BERT Q&A repo and select the *requirements.txt* file (Under \aistudio-samples\deep-learning-in-ais\bert_qa\).

11. Remove all versions of the libraries to download the latest for each (e.g. ==4.16.2).

12. Click **Create Project** to start the workspace and open the notebook.

### Deployment

1. Open the deployment.ipynb notebook.

2. Be sure you’re in the right kernel that you created, it will be called ***aistudio***.

3. From the menu, select ***Run***, then click **Run All**.

4. MLFlow hooks are already embedded, so this will register a model to MLFlow.

5. From the *Deployments* Tab, select **Deploy your model**.

6. Fill the information in the card and click **Deploy**.

7. Click the play icon.

8. When the URL endpoint is exposed, click on it.

9. At the top of the Swagger API page, click the demo link.

10. Congratulations, you have successfully created, registered, and locally deployed a BERT Q&A model!

## Other Demo Projects

- [NVIDIA RAPIDS Stocks](https://github.com/passarel/aistudio-ds-experiments/blob/main/rapids/cudf_pandas_stocks_demo.ipynb) [5 min]: This cuda Pandas notebook with NVIDIA’s RAPIDS Notebook container shows how you can accelerate Pandas data processing without changing how you code.

- [NGC NeMO Audio Translation](https://github.com/passarel/aistudio-ds-experiments/tree/main/Audio_Experiments/audio_translation) [10 min]: This audio text translation model uses two transformers, a HiFiGAN, and the NeMO Framework container to save, deploy, and locally publish an English to Spanish translation model.

- Try Your Own [15 min]: Create a project, connect data, clone a GitHub repo, get NVIDIA NGC models and containers, build a model, save to [MLFlow model registry](/docs/aistudio/using-aistudio/model-deployment/register-model-scikit-learn.md), and publish locally.
