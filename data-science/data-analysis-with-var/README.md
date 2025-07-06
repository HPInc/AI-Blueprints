# 😷 COVID Movement Patterns with VAR

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-supported-orange.svg?logo=jupyter)
![Pandas](https://img.shields.io/badge/Pandas-used-150458.svg?logo=pandas)

</div>

## 📚 Contents

- [🧠 Overview](#overview)
- [🗂 Project Structure](#project-structure)
- [⚙️ Setup](#setup)
- [🚀 Usage](#usage)
- [📞 Contact and Support](#contact-and-support)
---

# Overview

This project shows an visual data analysis of the effects of COVID-19 in two different cities: New York and London, using Vector Autoregression (VAR)

---

# Project Structure

```
├── docs/
│   └── swagger_UI_data_analysis_with_var.pdf                   # Swagger screenshot
│   └── swagger_UI_data_analysis_with_var.png                   # Swagger screenshot
├── notebooks
│   └── covid_movement_patterns_with_var.ipynb                  # Main notebook for the project              
├── README.md                                                   # Project documentation
```

---

# Setup

### 0 ▪ Minimum Hardware Requirements

Ensure your environment meets the minimum compute requirements for smooth image classification performance:

- **RAM**: 4 GB  

### 1 ▪ Create an AI Studio Project

- Create a new project in [Z by HP AI Studio](https://zdocs.datascience.hp.com/docs/aistudio/overview).

### 2 ▪ Set Up a Workspace

- Choose **Data Science** as the base image.

### 3 ▪ Download the Dataset
1. This experiment requires the **tutorial_data dataset** to run.
2. Download the dataset from `s3://dsp-demo-bucket/tutorial_data/` into an asset called **tutorial** and ensure that the AWS region is set to ```us-west-2```.

### 4 ▪ Clone the Repository

1. Clone the GitHub repository:  
   ```
   git clone https://github.com/HPInc/AI-Blueprints.git
   ```

2. Ensure all files are available after workspace creation.

---

# Usage

### 1 ▪ Run the Notebook

Execute the notebook inside the `notebooks` folder:

```bash
notebooks/covid_movement_patterns_with_var.ipynb
```

This will:

- Load and prepare the data
- Analyze the data Univariately and Bivariately
- Analyze the correlations between the features
- Decompose Time-Series
- Perform Exponential Smoothing Prediction Methods
- Perform Vector Autoregression (VAR)
- Test Cointegration
- Analyze Stationarity of a Time-Series
- Train the VAR model
- Analyze Autocorrelation of Residuals
- Forecast
- Evaluate the model
- Integrate MLflow 


### 2 ▪ Deploy the COVID Movement Patterns with VAR Service

- Go to **Deployments > New Service** in AI Studio.
- Name the service and select the registered model.
- Choose a model version and **GPU** it's **not necessary**.
- Choose the workspace.
- Start the deployment.
- Note: This is a local deployment running on your machine. As a result, if API processing takes more than a few minutes, it may return a timeout error. If you need to work with inputs that require longer processing times, we recommend using the provided notebook in the project files instead of accessing the API via Swagger or the web app UI.

### 3 ▪ Swagger / raw API

Once deployed, access the **Swagger UI** via the Service URL.

Paste a payload like:

```
{
  "inputs": {
    "city": [
      "New York"
    ],
    "steps": [
      2
    ]
  },
  "params": {}
}
```

And as response:

```
{
  "predictions": [
    {
      "retail_forecast": -33.728463233848125,
      "pharmacy_forecast": -30.846298674683787,
      "parks_forecast": 10.141207881911434,
      "transit_station_forecast": -22.428499788561272,
      "workplaces_forecast": -22.99469300562751,
      "case_count_forecast": 2408.7536947190256,
      "hospitalized_count_forecast": 98.59975487297108,
      "death_count_forecast": 11.877680770250993
    },
    {
      "retail_forecast": -38.21554995618252,
      "pharmacy_forecast": -30.56449061598022,
      "parks_forecast": -4.27564275594397,
      "transit_station_forecast": -39.14454598096768,
      "workplaces_forecast": -56.921910023079036,
      "case_count_forecast": 4896.93595036146,
      "hospitalized_count_forecast": 116.0842596188819,
      "death_count_forecast": 10.01707228744587
    }
  ]
}
```

---

# Contact and Support  

- Issues: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting. 

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**Z by HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
