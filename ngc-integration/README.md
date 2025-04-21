<h1 style="text-align: center; font-size: 40px;"> NGC Integration Sample Projects for Z by HP AI Studio </h1>

The sample projects in this folder demonstrate how to integrate **NVIDIA NGC (NVIDIA GPU Cloud)** resources with [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

We provide **four distinct sample projects**, each designed for quick and easy use to help you get started efficiently.

# 🎙️ Audio Translation with NeMo Models

This project demonstrates an end-to-end **audio translation pipeline** using **NVIDIA NeMo models**. It takes an English audio sample and performs:

1. **Speech-to-Text (STT)** conversion using Citrinet  
2. **Text Translation (TT)** from English to Spanish using NMT  
3. **Text-to-Speech (TTS)** synthesis in Spanish using FastPitch and HiFiGAN  

All steps are GPU-accelerated, and the full workflow is integrated with **MLflow** for experiment tracking and model registration.

# 📡 OpenCellID Exploratory Data Analysis with Panel and cuDF

This project is a GPU-accelerated, interactive **exploratory data analysis (EDA)** dashboard for the [OpenCellID](https://www.opencellid.org/) dataset. It uses **Panel** and **cuDF** to deliver lightning-fast geospatial analysis and visualization.

You can explore cell tower distributions by radio type, operator, country, and time window — rendered live on an interactive map with full GPU acceleration.

# 📈 Stock Analysis with Pandas and cuDF  

In this project, we provide notebooks to compare the execution time of dataset operations using traditional **Pandas** (CPU) versus **NVIDIA’s cuDF**, a GPU-accelerated drop-in replacement for Pandas. This example is presented in two different formats:

- **Original Example Notebook**: This version, created by NVIDIA, runs the entire evaluation within a single notebook. It includes downloading the data and restarting the kernel to activate the cuDF extension.

- **Data Analysis Notebooks**: These notebooks use preprocessed datasets of varying sizes from **datafabric** folder in AI Studio. The evaluation is split across two notebooks—one using Pandas (CPU) and the other using cuDF (GPU)—with performance metrics logged to **MLflow**.

# 🌍 Vacation Recommendation Agent

The Vacation Recommendation Agent is an AI-powered system designed to provide personalized travel recommendations based on user queries. It utilizes the NVIDIA NeMo Framework and BERT embeddings to generate relevant suggestions tailored to user preferences.

# Contact and Support  
- If you encounter issues, report them via GitHub by opening a new issue.  
- Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting.  

---

> Built with ❤️ using [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).