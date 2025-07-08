# DreamBooth Inference with Stable Diffusion 2.1

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-supported-orange.svg?logo=jupyter)
![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-2.1-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-used-ff6f00.svg?logo=tensorflow)
![DreamBooth](https://img.shields.io/badge/DreamBooth-fine--tuning-lightgreen.svg)

</div>

### Content
* [🧠 Overview](#overview)
* [🗂 Project Structure](#project-structure)
* [⚙️ Setup](#setup)
* [🚀 Usage](#usage)
* [📞 Contact and Support](#contact-and-support)

## Overview
This notebook performs image generation inference using the Stable Diffusion architecture, with support for both standard and DreamBooth fine-tuned models. It loads configuration and secrets from YAML files, enables local or deployed inference execution, and calculates custom image quality metrics such as entropy and complexity. The pipeline is modular, supports Hugging Face model loading, and integrates with advanced evaluation capabilities.

## Project Structure
```
├── config/                                     # Configuration files
│   ├── config.yaml                             # General settings (e.g., model config, mode)
│   └── secrets.yaml                            # API keys and credentials (e.g., HuggingFace)
│
├── core/                                        # Core Python modules
│       ├── custom_metrics/
│       │   └── image_metrics_scorers.py         # Image scoring (e.g., entropy, complexity)
│       ├── deploy/
│       │   └── deploy_image_generation.py       # Model deployment logic
│       ├── local_inference/
│       │   └── inference.py                     # Inference logic for standard Stable Diffusion
│       └── dreambooth_inference/
│           └── inference_dreambooth.py          # Inference for DreamBooth fine-tuned models
│
├── data/
│   └── img/                                     # Directory containing generated or input images
│       ├── 24C2_HP_OmniBook Ultra 14 i...       # Sample images used in inference
│       └── ...                                  # Other image files
│
├── docs/            
│     ├── Diagram dreambooth.png                 # Image generation diagram example  
├── notebooks/
│   ├── 04-image-generation_with_StableDiffusion.ipynb                          # Main notebook for running image generation inference
│
├── Diagram dreambooth.png                       # Diagram illustrating the DreamBooth architecture
├── README.md                                     # Project documentation
└── requirements.txt                              # Required dependencies
```

## Setup

### 0 ▪ Minimum Hardware Requirements

Ensure your environment meets the minimum hardware requirements for smooth model inference:

- RAM: 16 GB  
- VRAM: 8 GB  
- GPU: NVIDIA GPU


### Step 1: Create an AI Studio Project  
1. Create a **New Project** in AI Studio.   
2. (Optional) Add a description and relevant tags. 

### Step 2: Create a Workspace  
1. Select **Local GenAI** as the base image.
2. Upload the requirements.txt file and install dependencies.

### Step 3: Verify Project Files 
1. Clone the GitHub repository:  
   ```
   git clone https://github.com/HPInc/AI-Blueprints.git
   ```  
2. Make sure the folder `generative-ai/image-generation-with-stablediffusion` is present inside your workspace.

### Step 4: Use a Custom Kernel for Notebooks  
1. In Jupyter notebooks, select the **aistudio kernel** to ensure compatibility.


> ⚠️ **GPU Compatibility Notice**  
If you are using an older GPU architecture (e.g., **pre-Pascal**, such as **Maxwell or earlier**, like the GTX TITAN X), you may experience CUDA timeout errors during inference or training due to hardware limitations.  
To ensure stable execution, uncomment the line below at the beginning of your script or notebook to force synchronous CUDA execution:

```python
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
```

## Usage

### Step 1:
Run the following notebook `/image-generation_with_StableDiffusion.ipynb`:
1. Download the stabilityai/stable-diffusion-2-1 model from Hugging Face.
2. In the Training DreamBooth section of the notebook:
- Train your DreamBooth model (training time is approximately 1.5 to 2 hours).
- Monitor metrics using the **Monitor tab**, MLflow, and TensorBoard.

### Step 2:
1. After running the entire notebook, go to **Deployments > New Service** in AI Studio.
2. Create a service named as desired and select the **ImageGenerationService** model.
3. Choose a model version and enable **GPU acceleration**.
5. Deploy the service.
6. Once deployed, open the Service URL to access the Swagger API page.
7. How to use the API.

| Field               | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `prompt`           | Your input prompt                                                           |
| `use_finetuning`   | `True` to use your fine-tuned DreamBooth model, `False` for the base model |
| `height`, `width`  | Image dimensions                                                            |
| `num_images`       | Number of images to generate                                                |
| `num_inference_steps` | Number of denoising steps used by Stable Diffusion                       |

8. The API will return a base64-encoded image. You can convert it to a visual image using: https://base64.guru/converter/decode/image


## Contact and Support

- Issues & Bugs: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.


---

> Built with ❤️ using [**Z by HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html)
