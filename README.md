# Data Science with Z by HP AI Studio

This is a proposal for an initial structure of public repositories for educational material and demos. The main idea here is to make available a set of 15+ notebooks with end-to-end experiments split into subjects according to different topics. This way, we would have smaller repos (with no more than 5 experiments) - avoiding the current scenario of having to download a single big repo to run any experiment, but also without having too many different repos to give maintenance.


1. [Using AI Studio in 6 Steps](#first-course)
   1. [Projects, Workspaces and Github](#section1-1)
   2. [Datafabric](#section1-2)
   3. [Data visualization and monitoring](#section1-3)
   4. [Libraries and custom environments](#section1-4)
   5. [Deploying models locally](#section1-5)
   6. [Introducing CV and NLP](#section1-6)
2. [Deep Learning in AI Studio](#deep-learning-with-AI-Studio)
   1. [Image Classification](#section2-1)
   2. [Image Transformation: Super resolution](#section2-2)
   3. [Generating text by characters](#section2-3)
   4. [Introducing transformers for answering questions](#section2-4)
3. [Integrating with NGC](#using-NGC-resources)
   1. [Using RAPIDS to accelerate data processing](#section3-1)
   2. [Extending RAPIDS with data visualization](#section3-2)
   3. [NeMo for Audio and Text translation](#section3-3)
4. [Gen AI with Galileo and AIS](#using-prometheus)
   1. [Galileo Evaluate on RAG-based chatbot](#section4-1)
   2. [Improving chatbot quality with Galileo Observe and Protect ](#section4-2)
   3. [Summarizing text](#section4-3)
   4. [Code Generation](#section4-4)
   5. [Text Generation](#section4-5)

Below, we find a description of each specific subject/repository, as well as the intended demos/tutorials to be included on each one

<a id=first-course> </a>

## 1. Using AI Studio features in 6 steps 
* Currently saved on ai-studio fundamentals folder

This repo would have a different structure than the other ones. Five different notebooks would be used to illustrate different foundational features of AI Studio, in separate tutorials. These notebooks are:
  * Iris classification: One of the most traditional examples in ML, this notebook will be used to illustrate the most simple usage of AI Studio (section 1)
  * Movie experiment: This notebook is an example of a recommendation system, which can be used to show features as Data Fabric, ML Flow and Tensorboard monitoring and model deployment.
  * Tale of two cities: A nice example for different data visualization techniques, can also be used to demonstrate data fabric and installation of libraries/customization of environments
  * MNIST classification: End-to-end introdutory example of Computer Vision with AI Studio
  * Spam Classification: End-to-end introdutory example of Natural Language Processing with AI Studio

<a id=section1-1> </a>

### 1.1 Working with projects, workspaces and Github
#### Notebooks on this session
  * classification/iris
    * Needs to change the load_data, to use sklearn one 
#### Content
  * What is a project on AI Studio, and how does it work?
  * How to create a simple project?
  * How to add a simple Workspace inside a project (Minimal vs Data Science workspace)
  * How to connect to a Github Repository
  * How to access your notebook inside the workspace
  * What are the local folders?
  
  
<a id=section1-2> </a>

### 1.2 Using datafabric
#### Notebooks on this session
  * Introduce Movie experiment example
  * Introduce tale of two cities project
#### Content
  * How to add local folders to my project
  * How to access these local folders from inside the workspace
  * How to add cloud folders to my project
  * Why should you restart your workspace to access data fabric

<a id=section1-3> </a>

### 1.3 Data visualization and experiments monitoring
#### Notebooks on this session
  * Show data visualization in previous examples
  * Use movie experiment example to show monitoring
    * Can we change TB logging to use tensorboard library instead of TF
#### Content
  * Data visualization tools included
  * Using MLFlow to monitoring
  * Using Tensorboard to monitoring

<a id=section1-4> </a>

### 1.4 Installing libraries and configuring environments
#### Notebooks on this session
  * Use the same notebooks in previous sessions
    * Try to run them on minimal workspace, to show how to show the effects on environment
#### Content
  * Installing libraries with PIP
  * Custom workspaces/environments
  * Using conda environments manually

<a id=section1-5> </a>

### 1.5 Deploying models locally
#### Notebooks on this session
  * Use movie experiment example to show Model Service (make sure it works)
    * Create a quick UI later
#### Content
  * Logging and registering models in MLFlow
  * Deploying a service (swagger interface)
  * Adding a UI to the service 

<a id=section1-6> </a>

### 1.6 Introducing text and image processing
#### Notebooks on this session
  * MNIST (change Keras to scikit learn, so we do not use Tensorflow)
  * SpamClassification
#### Content
  * Use MNIST to show how to work with images
  * Use Spam classification to show how to work with text

### Extra Material
#### Notebooks on this session
  * Select in the future
#### Content
  * Briefly explain the extra notebooks
    
---

<a id=deep-learning-with-AI-Studio></a>

## 2. Deep Learning with Z by HP AI Studio

* Folder: deep-learning-in-ais

Starting in this second subject, each individual demo/tutorial is associated with a single notebook (and auxiliary files). In this section we will have 4 examples on how to use Tensorflow and Pytorch inside AI Studio, using GPU resources and our Deep Learning workspaces to easily put in practice to process images and language.

<a id=section2-1> </a>

### 2.1 Classifying images with TensorFlow/PyTorch
#### Notebooks on this session
* Basic Image Classification notebook
#### Content
* Use Deep Learning image to work with a Image Classification example
* Use Data from datafabric
* Ensure that MLFlow/Tensorboard are being used in the code
* Ensure that multiple runs are made, with different configurations, to allow comparison
* Ensure that GPU is being used

<a id=section2-2> </a>

### 2.2 Image transformation with Tensorflow/Pytorch (a different one from the previous session)
#### Notebooks on this session
* Super resolution example
#### Content
* Use Deep Learning image and the super resolution problem
* Use cloud data from Data Fabric
* Ensure that MLFlow/Tensorboard are being used
* Deploy a super resolution service with UI

<a id=section2-3> </a>

### 2.3 Generating text by character
#### Notebooks on this session
* Shakespeare example
#### Content
* Explain basic character generation using statistical patterns

<a id=section2-4> </a>

### 2.4 Simple Q&A with Bert
#### Notebooks on this session
* Bert QA
#### Content
* Explain basic usage of Hugging Face and transformers

---

<a id=using-NGC-resources></a>

## 3. Interating NVidia's NGC Resources with AI Studio

* Folder: ngc-integration

Here, we will aggregate the demos that use NGC resources, to show how to use them to our use cases

<a id=section3-1> </a>

### 3.1 Using Rapids to accelerate data processing
#### Notebooks on this session
* Rapids/Pandas Stock Demo
  
#### Content
* Show how Rapids can accelerate data operations done in pandas

<a id=section3-2> </a>

### 3.2 GeoProcessing with Rapids 
#### Notebooks on this session
* Rapids OpenCellID example

#### Content
* Expand Rapids acceleration to Data visualization of geo processing

<a id=section3-3> </a>

### 3.3 Using NeMo for audio and language processing
#### Notebooks on this session
* Audio translation examples

#### Content
* Nemo Framework image and how to use it in AI Studio
* Download models using NGC integration
* Running the models inside notebook
* Publishing a service using the models

---

<a id=using-prometheus></a>

## 4. Gen AI with AI Studio and Galileo

This actually is the same repository as the templates for [Prometheus](https://github.com/HPInc/aistudio-galileo-templates)

<a id=section4-1> </a>

### 4.1 General Chatbot with cloud model
#### Notebooks on this session
* Prometheus chatbot template
  
#### Content
* Creating a chatbot with langchain
* Using OpenAI model
* Evaluating experiment with Galileo Evaluate
* Using feedbacks from Galileo Evaluate to improve prompt

<a id=section4-2> </a>

### 4.2 Galileo Observe and Protect
#### Notebooks on this session
* Prometheus chatbot template
  
#### Content
* Instrumenting the code with Galileo Observe
* Monitoring the code with Galileo Observe interface
* Instrumenting the code with Galileo Protect
* Deploying the model locally
* Monitoring Galileo Protect errors and alerts

<a id=section4-3> </a>

### 4.3 Summarization with local model
#### Notebooks on this session
* Prometheus summarization template
  
#### Content
* Creating a custom pipeline for summarization
* Using multiple data connectors
* Using locally deployed model
* Custom chains on Galileo Evaluate
* Custom scorers on Galileo Evaluate
* Deploying the service and adding Observe and Protect

<a id=section4-4> </a>

### 4.4 Code Generation with AI Studio and Galileo
#### Notebooks on this session
* Prometheus code generation example
  
#### Content
* Explain the content of this example

<a id=section4-5> </a>

### 4.5 Text Generation with AI Studio and Galileo
#### Notebooks on this session
* Prometheus text generation example
  
#### Content
* Explain the content of this example
