# Interactive ORPO Fine-Tuning & Inference Hub for Open LLMs

## 📚 Contents

- Overview
- Project Structure
- Setup
- Usage
- Contact & Support

---

## 🧠 Overview
This project demonstrates a full-stack LLM fine-tuning experiment using ORPO (Open-Source Reinforcement Pretraining Objective) to align a base language model with human preference data. It leverages the **Z by HP AI Studio Local GenAI environment**, and uses models such as LLaMA 3, Gemma 1B, and Mistral 7B as foundations.

We incorporate:

- **Galileo PromptQuality** for evaluating model responses with human-like scorers (e.g., context adherence)
- **TensorBoard** for human feedback visualization before fine-tuning
- A flexible model selector and inference runner architecture
- A comparative setup to benchmark base vs fine-tuned models on the same prompts

---

## 🗂 Project Structure

```
├── notebooks
│   ├── config
│   │   ├── default_config_cpu.yaml
│   │   ├── default_config_multi-gpu.yaml
│   │   └── default_config_one-gpu.yaml
│   ├── core
│   │   ├── comparer
│   │   │   └── galileo_hf_model_comparer.py
│   │   ├── data_visualizer
│   │   │   └── feedback_visualizer.py
│   │   ├── deploy
│   │   │   └── deploy_fine_tuning.py
│   │   ├── local_inference
│   │   │   └── inference.py
│   │   ├── selection
│   │   │   └── model_selection.py
│   │   └── target_mapper
│   │       └── lora_target_mapper.py
│   └── fine_tuning_orpo.ipynb
├── README.md
└── requirements.txt

```
---

## ⚙️ Setup

### Step 0: Minimum Hardware Requirements
To ensure smooth execution and reliable model deployment, make sure your system meets the following **minimum hardware specifications** based on the selected model and task (inference or fine-tuning):

### ✅ Model Hardware Matrix

| **Model**                                    | **Task**       | **Min VRAM** | **Min RAM** | **GPU Recommendation**              |
|---------------------------------------------|----------------|--------------|-------------|--------------------------------------|
| `mistralai/Mistral-7B-Instruct-v0.1`         | Inference      | 12 GB        | 32 GB       | RTX 3080, A100 (for 4-bit QLoRA)     |
|                                             | Fine-tuning    | 40–48+ GB    | 64+ GB      | RTX 4090, A100, H100                 |
| `meta-llama/Llama-2-7b-chat-hf`              | Inference      | 12 GB        | 32 GB       | RTX 3080 or better                   |
|                                             | Fine-tuning    | 40–48+ GB    | 64+ GB      | RTX 4090+                            |
| `meta-llama/Meta-Llama-3-8B-Instruct`        | Inference      | 16 GB        | 32 GB       | RTX 3090, 4090                       |
|                                             | Fine-tuning    | 64+ GB       | 64–96 GB    | Dual RTX 4090 or A100                |
| `google/gemma-7b-it`                         | Inference      | 12 GB        | 32 GB       | RTX 3080 or better                   |
|                                             | Fine-tuning    | 40+ GB       | 64 GB       | RTX 4090                             |
| `google/gemma-3-1b-it`                       | Inference      | 8 GB         | 16–24 GB    | RTX 3060 or better                   |
|                                             | Fine-tuning    | 16–24 GB     | 32–48 GB    | RTX 3080 / 3090                      |



> ⚠️ These recommendations are based on community benchmarks and documentation provided by Hugging Face, Unsloth, and Google. For production workloads, always monitor VRAM/RAM usage on your system.

### Step 1: Create an AI Studio Project

- Create a new project in [Z by HP AI Studio](https://zdocs.datascience.hp.com/docs/aistudio/overview).

### Step 2: Set Up a Workspace

- Choose **Local GenAI** as the base image.

### Step 3: Clone the Repository

```bash
https://github.com/HPInc/aistudio-galileo-templates.git
```

- Ensure all files are available after workspace creation.

### Step 4: Configure Secrets and Paths

- Add your API keys to the `secrets.yaml` file located in the `configs` folder:
  - `HUGGINGFACE_API_KEY`: Required to use Hugging Face-hosted models instead of a local LLaMA model.
  - `GALILEO_API_KEY`: Required to connect to Galileo for evaluation, protection, and observability features.
- Edit `config.yaml` with relevant configuration details.


---

## 🚀 Usage

### Step 1: Run the Notebook

Execute the notebook inside the `notebooks` folder:

```bash
notebooks/fine_tuning_orpo.ipynb
```

This will:

- Select and download a compatible model from Hugging Face
- Apply QLoRA configuration and prepare the model for training
- Run the fine-tuning using ORPO
- Perform evaluation and comparison with the base model using Galileo Prompt Quality
- Register and serve both base and fine-tuned models via MLflow

### Step 2: Deploy the Chatbot Service

-  Go to **Deployments > New Service** in AI Studio.
- Name the service and select the registered model.
- Choose a model version and enable **GPU acceleration**.
- Start the deployment.
- Once deployed, access the **Swagger UI** via the Service URL.

---

## 📞 Contact & Support

- 💬 For issues or questions, please [open a GitHub issue](https://github.com/HPInc/aistudio-galileo-templates/issues).
- 📘 Refer to the official [AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview) for detailed instructions and troubleshooting tips.

---

> Built with ❤️ using Z by HP AI Studio.
