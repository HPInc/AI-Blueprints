<h1 style="text-align: center; font-size: 45px;"> AI Blueprint Projects for Z by HP AI Studio 🚀 </h1>

# Content  
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data Science and Machine Learning](#data-science-and-machine-learning)
- [Deep Learning](#deep-learning)
- [Generative AI](#generative-ai)
- [NVIDIA GPU Cloud](#nvidia-gpu-cloud)
- [Contact and Support](#contact-and-support)

# Overview 

This repository contains a collection of sample projects that you can run quickly and effortlessly, designed to integrate seamlessly with [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview). Each project runs end-to-end, offering out-of-the-box, ready-to-use solutions across various domains, including data science, machine learning, deep learning, and generative AI.  

The projects leverage local open-source models such as **LLaMA** (Meta), **BERT** (Google), and **CitriNet** (NVIDIA), alongside selected online models accessible via **Hugging Face**. These examples cover a wide range of use cases, including **data visualization**, **stock analysis**, **audio translation**, **agentic RAG applications**, and much more.  

We are continuously expanding this collection with new projects. If you have suggestions or would like to see a specific sample project integrated with [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview), please feel free to open a new issue in this repository — we welcome your feedback!

# Repository Structure 

- ai-studio-fundamentals
  - Iris flowers classification.ipynb
  - Recommender Systems.ipynb
  - Spam Detection and NLP.ipynb
  - [MLFlow] MNIST with Keras.ipynb
  - a-tale-of-two-cities-analyzing-trends.ipynb
- deep-learning-in-ais
  - bert_qa
  - super_resolution
  - text_generation
- gen-ai
  - agentic_rag_llama
- ngc-integration
  - audio_translation_with_nemo_models
  - opencellid_eda_with_panel_and_cuDF
  - stock_analysis_with_pandas_and_cuDF
  - vacation_recommendation_agent_with_bert

# Data Science and Machine Learning

The sample projects in this folder demonstrate how to build data science and machine learning applications with [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

We provide **5 sample projects**, each designed for quick and easy use to help you get started efficiently.

### 🌸 Iris Flower Classification

This project is a simple **classification** experiment focused on predicting species of **Iris flowers**.  

It runs on the **Data Science Workspace**, demonstrating basic supervised learning techniques for multi-class classification tasks.

### 🖌️ MNIST Handwritten Digit Classification

This project performs basic **image classification** using the **TensorFlow** framework.  

It trains a model to classify handwritten digits from the **MNIST** dataset and runs on the **Deep Learning Workspace**.

### 🏙️ A Tale of Two Cities: Mobility Regression During COVID-19

This project explores a **regression** experiment using **mobility data** collected during the COVID-19 pandemic.  

It highlights how city-level movement patterns changed during the crisis. The experiment runs on the **Data Science Workspace**.

### 🎬 Movie Recommender System with TensorFlow

This project builds a simple **recommender system** for movies using **TensorFlow**.  

It trains on user-item interaction data to predict movie preferences and runs on the **Deep Learning Workspace**.

### 🚫 Spam Detection with Deep Learning

This project implements a **text classification** system to detect **spam** messages.  

It uses deep learning techniques and requires the **Deep Learning Workspace** for training and inference.




# Deep Learning

The sample projects in this folder demonstrate how to build deep learning applications with [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

We provide **3 sample projects**, each designed for quick and easy use to help you get started efficiently.

### 🧠 BERT Question Answering with MLflow

This project demonstrates a simple **BERT Question Answering (QA)** experiment. It provides code to train a BERT-based model, as well as instructions to load a pretrained model from **Hugging Face**.  

The model is deployed using **MLflow** to expose an inference service capable of answering questions based on input text.

### 🖼️ Image Super Resolution with Convolutional Networks

This project showcases a **Computer Vision** experiment that applies convolutional neural networks for **image super-resolution** — enhancing the quality and resolution of input images.  

### ✍️ Character-Level Text Generation with Shakespeare Dataset

This project illustrates how to build a simple **character-by-character text generation** model.  

It trains on a dataset containing **Shakespeare's texts**, demonstrating the fundamentals of text generation by predicting one character at a time.

# Generative AI

The sample projects in this folder demonstrate how to build generative AI applications with [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

We provide **1 sample project**, each designed for quick and easy use to help you get started efficiently.

### 🤖 Agentic RAG Notebook with Llama 2 and ChromaDB

This project implements an **Agentic Retrieval-Augmented Generation (RAG)** pipeline combining **Llama 2** and **ChromaDB**.  

It features an intelligent question-answering system where the model dynamically decides whether external document context is needed before responding, ensuring highly accurate and contextually relevant answers through an agentic workflow.


# NVIDIA GPU Cloud

The sample projects in this folder demonstrate how to integrate **NVIDIA NGC (NVIDIA GPU Cloud)** resources with [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

We provide **four distinct sample projects**, each designed for quick and easy use to help you get started efficiently.

### 🎙️ Audio Translation with NeMo Models

This project demonstrates an end-to-end **audio translation pipeline** using **NVIDIA NeMo models**. It takes an English audio sample and performs:

1. **Speech-to-Text (STT)** conversion using Citrinet  
2. **Text Translation (TT)** from English to Spanish using NMT  
3. **Text-to-Speech (TTS)** synthesis in Spanish using FastPitch and HiFiGAN  

All steps are GPU-accelerated, and the full workflow is integrated with **MLflow** for experiment tracking and model registration.

### 📡 OpenCellID Exploratory Data Analysis with Panel and cuDF

This project is a GPU-accelerated, interactive **exploratory data analysis (EDA)** dashboard for the [OpenCellID](https://www.opencellid.org/) dataset. It uses **Panel** and **cuDF** to deliver lightning-fast geospatial analysis and visualization.

You can explore cell tower distributions by radio type, operator, country, and time window — rendered live on an interactive map with full GPU acceleration.

### 📈 Stock Analysis with Pandas and cuDF  

In this project, we provide notebooks to compare the execution time of dataset operations using traditional **Pandas** (CPU) versus **NVIDIA’s cuDF**, a GPU-accelerated drop-in replacement for Pandas. This example is presented in two different formats:

- **Original Example Notebook**: This version, created by NVIDIA, runs the entire evaluation within a single notebook. It includes downloading the data and restarting the kernel to activate the cuDF extension.

- **Data Analysis Notebooks**: These notebooks use preprocessed datasets of varying sizes from **datafabric** folder in AI Studio. The evaluation is split across two notebooks—one using Pandas (CPU) and the other using cuDF (GPU)—with performance metrics logged to **MLflow**.

### 🌍 Vacation Recommendation Agent with NeMo and BERT

This project implements an **AI-powered recommendation agent** that delivers personalized travel suggestions based on user queries. 

It leverages the **NVIDIA NeMo Framework** and **BERT embeddings** to understand user intent and generate highly relevant, tailored vacation recommendations.



# Contact and Support  
- If you encounter issues, report them via GitHub by opening a new issue.  
- Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting.  

---

> Built with ❤️ using [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).
