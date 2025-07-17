<h1 style="text-align: center; font-size: 40px;"> NGC Integration Blueprint Projects for HP AI Studio </h1>

The sample projects in this folder demonstrate how to integrate **NVIDIA NGC (NVIDIA GPU Cloud)** resources with [**HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

We provide **4 blueprint projects**, each designed for quick and easy use to help you get started efficiently.

## Repository Structure

The repository is organized into the following structure:

```
├── audio_translation_with_nemo_models/
│    ├── data/                                                                    # Data assets used in the project   
│    ├── ForrestGump.mp3
│       └── June18.mp3
│    ├── demo                                                                    # UI-related files
│       └── ...
│    ├── docs
│       ├── react_ui_for_audio_translation.png                                  # React UI screenshot 
│       ├── streamlit_ui_for_audio_translation.png                              # Streamlit UI screenshot screenshot    
│       ├── successful react ui result for audio translation.pdf                # React UI screenshot 
│       └── successful streamlit ui result for audio translation. pdf           # Streamlit UI screenshot
├── opencellid_eda_with_panel_and_cuDF/                   
│    ├── docs/
│    │   └── ui_opencellid.png                                   # opencellid UI screenshot
│    ├── notebooks/
│    │   └── opencellid_eda_with_panel_and_cuDF.ipynb            # Main notebook for the project
│    ├── src/                                                    # Core Python modules
│    │   └── opencellid_downloader.py
│    ├── README.md                                               # Project documentation
│    └── requirements.txt                                        # Python dependencies (used with pip install)
│
├── stock_analysis_with_pandas_and_cuDF/                     
│    ├── notebooks/                                              # Main notebooks for the project
│    │   ├── stock_analysis_with_pandas.ipynb
│    │   └── stock_analysis_with_pandas_and_cuDF.ipynb
│    ├── README.md                                               # Project documentation
│    └── requirements.txt                                        # Python dependencies (used with pip install)
│
├── vacation_recommendation_agent_with_bert/
│    ├── data/                                                   # Data assets used in the project
│    │   └── raw/
│    │       └── corpus.csv
│    ├── demo/                                                   # UI-related files
│    │   └── index.html
│    ├── docs/
│    │   ├── architecture.md                                     # Model Details and API Endpoints
│    │   └── ui_vacation.png                                     # UI screenshot
│    ├── notebooks/                                              # Main notebooks for the project
│    │   ├── 00_Word_Embeddings_Generation.ipynb
│    │   └── 01_Bert_Model_Registration.ipynb
│    ├── README.md                                               # Project documentation
│    └── requirements.txt                                        # Python dependencies (used with pip install)



```

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

The **Vacation Recommendation Agent** is an AI-powered system designed to provide personalized travel recommendations based on user queries. 

It utilizes the **NVIDIA NeMo Framework** and **BERT embeddings** to generate relevant suggestions tailored to user preferences.  


# Contact and Support

- Issues: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting.

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
