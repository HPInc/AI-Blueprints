# Turbofan Engine RUL Prediction

This repository contains notebooks and scripts for predicting the Remaining Useful Life (RUL) of turbofan engines using a Recurrent Neural Network (RNN) model. The project is inspired by the work on Weibull Time-To-Event Recurrent Neural Networks (WTTE-RNN) for churn modeling, as detailed in [this blog post](https://ragulpr.github.io/2016/12/22/WTTE-RNN-Hackless-churn-modeling/).

## Notebooks

### 1. `data_fetch.ipynb`

This notebook is responsible for downloading and preparing the dataset required for training and testing the RNN model. It performs the following tasks:

- Downloads the Turbofan Engine Degradation Simulation Data Set from a specified URL.
- Extracts the downloaded zip file and organizes the data into a structured directory.
- Lists the files in the target directory to ensure the data is ready for use.

### 2. `Turbofan_v3.ipynb`

This notebook implements the RNN model for predicting the RUL of turbofan engines. It includes the following sections:

- **Data Loading and Preparation**: Loads the multivariate time series data, fills missing values, normalizes sensor measurements, and computes the time to failure for each engine unit.
- **Model Definition**: Defines a PyTorch model class `WTTEModel` using LSTM layers to predict Weibull distribution parameters.
- **Loss Function**: Implements the WTTE loss function to train the model.
- **Data Preparation for PyTorch**: Prepares the data for training and testing using PyTorch's `Dataset` and `DataLoader`.
- **Training Loop**: Trains the model using the Adam optimizer, logging training and test losses over epochs.
- **Visualization**: Provides enhanced visualizations of predictions vs. actual targets, training and test losses, and Weibull distribution plots for a random engine.

## Requirements

To run the notebooks, you need to have the following Python packages installed:

- pandas
- matplotlib
- torch
- numpy
- requests
- zipfile
- shutil

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. **Download and Prepare Data**: Run the `data_fetch.ipynb` notebook to download and prepare the dataset.

2. **Train and Evaluate the Model**: Open and execute the `Turbofan_v3.ipynb` notebook to train the RNN model and evaluate its performance on the test data. The notebook includes detailed instructions and code cells for each step of the process.

3. **Visualize Results**: Use the visualization sections in the `Turbofan_v3.ipynb` notebook to analyze the model's predictions and performance metrics.

## Inspiration

This project is inspired by the WTTE-RNN approach for churn modeling, which can be adapted for various time-to-event prediction tasks, including predicting the RUL of machinery. For more details, refer to the [WTTE-RNN blog post](https://ragulpr.github.io/2016/12/22/WTTE-RNN-Hackless-churn-modeling/).
