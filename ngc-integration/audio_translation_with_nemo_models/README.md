# 🎙️ Audio Translation with NeMo Models

## 📚 Contents

- Overview  
- Project Structure  
- Setup  
- Usage  
- Contact & Support

---

## 🧠 Overview

This project demonstrates an end-to-end **audio translation pipeline** using **NVIDIA NeMo models**. It takes an English audio sample and performs:

1. **Speech-to-Text (STT)** conversion using Citrinet  
2. **Text Translation (TT)** from English to Spanish using NMT  
3. **Text-to-Speech (TTS)** synthesis in Spanish using FastPitch and HiFiGAN  

All steps are GPU-accelerated, and the full workflow is integrated with **MLflow** for experiment tracking and model registration.

---

## 🗂 Project Structure

```
├── data
│   └── ForrestGump.mp3
│   └── June18.mp3
├── demo
│   └── ...
├── notebooks
│   └── english_to_chinese.ipynb
│   └── english_to_spanish.ipynb
│   └── french_to_english.ipynb
├── src
│   └── deploy_ms.py
├── README.md
└── requirements.txt
```

---

## ⚙️ Setup

### Step 0: Minimum Hardware Requirements

Ensure your environment meets the minimum hardware requirements for smooth model inference:

- RAM: 16 GB  
- VRAM: 8 GB  
- GPU: NVIDIA GPU

### Step 1: Create an AI Studio Project

- Create a new project in [Z by HP AI Studio](https://zdocs.datascience.hp.com/docs/aistudio/overview).

### Step 2: Set Up a Workspace

- Choose the **NeMo Framework** image from the **NVIDIA NGC Catalog** during workspace setup.

### Step 3: Clone the Repository

```bash
https://github.com/HPInc/aistudio-samples.git
```

- Ensure all files are available after workspace creation.

### Step 4: Add Required NeMo Models

From the **Models** tab, add the following models from the **NVIDIA NGC Catalog**:

1. **Speech-to-Text (STT)**  
   - Model: `stt_en_citrinet_1024_gamma_0_25-1.0.0`  
   - Asset Name: `STT En Citrinet 1024 Gamma 0.25`

2. **Neural Machine Translation (NMT)**  
   - Model: `nmt_en_es_transformer12x2-1.0.0rc1`  
   - Asset Name: `NMT En Es Transformer12x2`

3. **Text-to-Speech (TTS)**  
   - Model: `tts_es_multispeaker_fastpitchhifigan-1.15.0`  
   - Asset Name: `TTS Es Multispeaker FastPitch HiFiGAN`

Make sure these models are downloaded and available in the `datafabric` folder inside your workspace.

---

## 🚀 Usage

### Step 1: Run the Notebook

Open and run the notebook located at:

```bash
notebooks/english_to_spanish.ipynb
```

This will:

- Load STT, NMT, and TTS models from the NGC assets  
- Convert an English audio file to English text  
- Translate the text into Spanish  
- Synthesize spoken Spanish audio from the translated text  
- Log the entire workflow as a composite model in **MLflow**

> Note: If your asset names differ from the recommended ones, update the notebook accordingly. Model loading may take time and can fail on GPUs with <8GB VRAM.

### Step 2: Register and Deploy the Model

- After running the notebook, the composite model will be registered in **MLflow**.  
- To deploy the service, run the following script in the terminal:

```bash
python src/deploy_ms.py
```
or

- Confirm that the model is registered in `Monitor > MlFlow > Models`  
- Go to `Deployments > New Service`  
- Select the registered model and enable GPU acceleration  
- Once deployed, click the **Play** button to launch the Swagger UI for inference

---

## 📞 Contact & Support

- 💬 For issues or questions, please [open a GitHub issue](https://github.com/HPInc/aistudio-samples/issues).  
- 📘 Refer to the official [AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview) for setup and troubleshooting guidance.

---

> Built with ❤️ using [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).