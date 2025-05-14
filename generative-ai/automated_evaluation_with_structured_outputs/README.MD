# 📊 Automated Evaluation with Structured Outputs

# 📚 Contents

* [🧠 Overview](#overview)
* [🗂 Project Structure](#project-structure)
* [⚙️ Setup](#setup)
* [🚀 Usage](#usage)
* [📞 Contact and Support](#contact-and-support)

---

# Overview

**Automated Evaluation with Structured Outputs** turns a local **Meta‑Llama‑3** model into an MLflow‑served scorer that rates any batch of texts (e.g., ISEF abstracts) against arbitrary rubric criteria.
The pipeline:

* Generates scores locally via `llama.cpp` (no data leaves your machine)
* Registers the evaluator as a **pyfunc** model in MLflow
* Exposes a REST `/invocations` endpoint
* Ships two front‑ends — a **Streamlit** dashboard and a pure **HTML/JS** UI — for instant human‑friendly interaction and CSV download.

---

# Project Structure

```
├── core
│   └── llama_evaluator
│       ├── model.py               # LlamaEvaluatorModel class (MLflow)
│       └── streamlit_app.py       # Streamlit UI
├── demo
│   ├── index.html                 # Lightweight HTML/JS UI
│   └── assets/
├── notebooks
│   └── build_and_register.ipynb   # One‑click notebook to train & log model
├── datafabric/Meta‑Llama‑3‑8B‑…   # Quantized GGUF model
├── docs/
│   └── ui_llamascore.png          # UI screenshot
├── README.md
└── requirements.txt
```

---

# Setup

### 0 ▪ Minimum hardware

* **RAM:** 32 GB
* **VRAM:** 10 GB (16 GB+ for fastest latency)
* **GPU:** NVIDIA (CUDA)

### 1 ▪ Create an AI Studio project

Log into [Z by HP AI Studio](https://zdocs.datascience.hp.com/docs/aistudio/overview) and start a **Local GenAI** workspace.

### 2 ▪ Clone the repo

```bash
git clone https://github.com/<your‑org>/automated‑evaluation‑structured‑outs.git
```

### 3 ▪ Add the Llama‑3 model

Download the 8‑B‑Instruct GGUF via Models tab:

* **Model name:** `Meta‑Llama‑3‑8B‑Instruct‑Q8_0`
* **Source:** AWS S3 → `s3://…/Meta-Llama-3-8B-Instruct-Q8_0.gguf`

Place it under `datafabric/`.

### 4 ▪ Configure secrets (optional)

Edit `configs/secrets.yaml` if you plan to call external APIs; not required for pure local inference.

---

# Usage

### 1 ▪ Run the notebook

Open **`notebooks/build_and_register.ipynb`**, run all cells.
This will:

1. Load the GGUF model
2. Log **`LlamaEvaluatorModel`** to MLflow
3. Serve it locally on port **5000**

### 2 ▪ Launch the Streamlit UI

```bash
python core/llama_evaluator/streamlit_app.py
# or
streamlit run core/llama_evaluator/streamlit_app.py
```

Navigate to the shown URL, upload a CSV, tweak parameters, and view scores.

### 3 ▪ Use the HTML UI

Serve `demo/index.html` from the same origin as your MLflow service (e.g. drop it in the model‑serve folder).
The page auto‑detects the host and calls **`/invocations`**.

### 4 ▪ Swagger / raw API

If you enabled **MLflow deployments**, open Swagger at

```
http://<host>:5000/docs#/default/invocations_invocations_post
```

Paste a payload like:

```jsonc
{
  "dataframe_split": {
    "columns": ["BoothNumber", "AbstractText"],
    "data": [
      ["TEST001","Microplastics impact on marine life"],
      ["TEST002","Low‑cost solar charger for off‑grid use"]
    ]
  },
  "params": {
    "key_column":"BoothNumber",
    "eval_column":"AbstractText",
    "criteria":"[\"Originality\",\"ScientificRigor\",\"Clarity\",\"Relevance\",\"Feasibility\",\"Brevity\"]",
    "batch_size":2
  }
}
```

### Successful UI demo

![LlamaScore UI](docs/ui_llamascore.png)

---

# Contact and Support

* 💬 Open an [issue](https://github.com/<your‑org>/automated‑evaluation‑structured‑outs/issues) for bugs or questions.
* 📘 See [AI Studio Docs](https://zdocs.datascience.hp.com/docs/aistudio/overview) for workspace help.

---

> Built with ❤️ using [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).
