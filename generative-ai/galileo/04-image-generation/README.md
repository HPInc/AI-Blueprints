# DreamBooth Inference with Stable Diffusion 2.1

### Content
- Overview
- Project Structure
- Setup
- Usage
- Contact and support

## Overview
This notebook performs image generation inference using the Stable Diffusion architecture, with support for both standard and DreamBooth fine-tuned models. It loads configuration and secrets from YAML files, enables local or deployed inference execution, and calculates custom image quality metrics such as entropy and complexity. The pipeline is modular, supports Hugging Face model loading, and integrates with PromptQuality for evaluation and tracking.

## Project Structure
```
├── data/
│   └── img/                                     # Directory containing generated or input images
│       ├── 24C2_HP_OmniBook Ultra 14 i...       # Sample images used in inference
│       └── ...                                  # Other image files
│
├── notebooks/
│   ├── inference.ipynb                          # Main notebook for running image generation inference
│   ├── config/                                  # Configuration files
│   │   ├── config.yaml                          # General settings (e.g., model config, mode)
│   │   └── secrets.yaml                         # API keys and credentials (e.g., HuggingFace, Galileo)
│   └── core/                                    # Core Python modules
│       ├── custom_metrics/
│       │   └── image_metrics_scorers.py         # Image scoring (e.g., entropy, complexity)
│       ├── deploy/
│       │   └── deploy_image_generation.py       # Model deployment logic
│       ├── local_inference/
│       │   └── inference.py                     # Inference logic for standard Stable Diffusion
│       └── dreambooth_inference/
│           └── inference_dreambooth.py          # Inference for DreamBooth fine-tuned models
│
├── Diagram dreambooth.png                       # Diagram illustrating the DreamBooth architecture
├── README.md                                     # Project documentation
└── requirements.txt                              # Required dependencies
```

## Setup

### Step 1: Create an AI Studio Project  
1. Create a **New Project** in AI Studio.   
2. (Optional) Add a description and relevant tags. 

### Step 2: Create a Workspace  
1. Select **Local GenAI** as the base image.
2. Upload the requirements.txt file and install dependencies.

### Step 3: Verify Project Files 
1. Clone the GitHub repository:  
   ```
   git clone https://github.com/HPInc/aistudio-samples.git
   ```  
2. Make sure the folder `aistudio-samples/04-image-generation` is present inside your workspace.

### Step 4: Use a Custom Kernel for Notebooks  
1. In Jupyter notebooks, select the **aistudio kernel** to ensure compatibility.

## Usage

### Step 1:
Run the following notebook `/inference.ipynb`:
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
- If you encounter issues, report them via GitHub by opening a new issue.  
- Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting.
