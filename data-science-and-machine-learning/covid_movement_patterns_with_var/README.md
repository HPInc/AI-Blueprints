# 😷 COVID Movement Patterns with VAR

## 📚 Contents

- [🧠 Overview](#overview)
- [🗂 Project Structure](#project-structure)
- [⚙️ Setup](#setup)
- [🚀 Usage](#usage)
- [📞 Contact and Support](#contact-and-support)
---

## 🧠 Overview

This project shows an visual data analysis of the effects of COVID-19 in two different cities: New York and London, using Vector Autoregression (VAR)

---

## 🗂 Project Structure

```
├── notebooks
│   └── covid_movement_patterns_with_var.ipynb                  # Main notebook for the project              
├── README.md                                                   # Project documentation
```

---

## ⚙️ Setup

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

## 🚀 Usage

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


---

## 📞 Contact & Support

- Issues & Bugs: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.


---

> Built with ❤️ using [**Z by HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).