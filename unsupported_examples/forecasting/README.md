# **Forecasting with PyTorch**
---
> Forecasting experiments with `torch`, `TensorFlow` , and the main frameworks used for this task.

----------

## Experiment details

### **RNN with Time Series TF** 

Experiment to forecast sales using LSTM (Long Short-Term Memory) neural networks within the TensorFlow and Keras frameworks, employing time series data processing and model training techniques.

- **Required resources** 

    Optimal for execution on GPU due to LSTM's computational demands, but can also run on CPU. Requires moderate to high memory space depending on the dataset size and model complexity.

- **Dataset details** 

    The dataset is a CSV file composed by 326 rows and 2 columns. The file has no more than 5.18 KB.

---

### **RNN with Time Series Torch** 

This experiment utilizes a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) cells to forecast alcohol sales based on historical data. The main goal is to predict future sales figures using time series analysis, employing deep learning techniques in PyTorch.

- **Required resources** 

    Optimal for execution on GPU due to LSTM's computational demands, but can also run on CPU. Requires moderate to high memory space depending on the dataset size and model complexity.

- **Dataset details** 

    The dataset is a CSV file composed by 326 rows and 2 columns. 

---

### **A tale of two cities analyzing trends** 

This experiment analyzes and forecasts mobility and COVID-19 case trends in New York and London. Using advanced statistical models like Vector Autoregression (VAR) and Exponential Smoothing, the study integrates data on mobility changes from sources like Google’s COVID-19 Community Mobility Reports with public health data on COVID-19 cases.

- **Required resources** 

    The study can be performed on both CPU and GPU. While it is computationally intensive due to the complexity of VAR models and large datasets.

- **Dataset details** 

    The datasets include:
    - New York Mobility Data: 53,803 rows and 15 columns detailing trends at locations like parks, transit stations, and retail areas.
    - London Mobility Data: 29,649 rows and 15 columns capturing similar mobility trends as the New York dataset.
    - New York COVID-19 Case Data: 856 rows and 64 columns with detailed public health data including daily cases, hospitalizations, and deaths.
    - London COVID-19 Case Data: 839 rows and 22 columns providing comprehensive health metrics similar to the New York health data.

---

### **General forecasting sklearn statsmodel** 

This experiment develops a model for forecasting sales across various departments in retail stores, leveraging statistical techniques like linear regression enhanced with Fourier series for seasonality, and including data-driven features from promotional and transactional activities.

- **Required resources** 

    The analysis can be run on both CPU and GPU.

- **Dataset details** 

    The datasets integrated into this analysis include:
    - Holiday Events Data: 352 rows and 6 columns detailing national and regional holidays which are used to adjust the sales forecast around significant dates.
    - Sales Data: Extensive sales records from retail stores, comprising 1,048,576 rows and 6 columns, showing daily sales figures for different departments, allowing for granular analysis and forecasting.
    - Transaction Data: Contains 83,489 rows and 3 columns, providing daily transaction counts which are used to understand customer flow and its impact on sales.    

---

### **Seasonal forecasting sklearn statsmodel** 

This experiment focuses on forecasting sales for stores within Cluster 13, incorporating data-driven adjustments for seasonal trends, holidays, and promotional activities. It utilizes linear regression with Fourier series to capture seasonality and additional features that reflect sales dynamics on holidays and promotional days.

- **Required resources** 

    The analysis can be run on both CPU and GPU.

- **Dataset details** 

    The datasets integrated into this analysis include:
    - Holiday Events Data: 352 rows and 6 columns detailing national and regional holidays which are used to adjust the sales forecast around significant dates.
    - Sales Data: Extensive sales records from retail stores, comprising 1,048,576 rows and 6 columns, showing daily sales figures for different departments, allowing for granular analysis and forecasting.
    - Transaction Data: Contains 83,489 rows and 3 columns, providing daily transaction counts which are used to understand customer flow and its impact on sales.    

