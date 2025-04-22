# Stock and Cryptocurrency Movement Simulation

This repository contains Python code for simulating and analyzing the price movements of stocks and cryptocurrencies. The simulation uses stochastic models to generate synthetic price data and provides tools for statistical analysis and visualization.

## Overview

Financial markets exhibit complex behaviors that can be modeled using mathematical techniques. This project implements a correlated geometric Brownian motion model to simulate the price movements of two asset classes - traditional stocks and cryptocurrencies - while maintaining realistic statistical properties and correlation structures.

## Features

- **Synthetic Data Generation**: Create realistic price series using geometric Brownian motion
- **Correlated Asset Modeling**: Simulate assets with specified correlation coefficients
- **Statistical Analysis**: Calculate key metrics like returns, volatility, Sharpe ratios, and drawdowns
- **Visualization Tools**: Plot price movements, return distributions, and regression analyses
- **Configurable Parameters**: Easily adjust return expectations, volatility, correlation, and time horizon

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy

## Installation

```bash
git clone https://github.com/yourusername/stock-crypto-simulation.git
cd stock-crypto-simulation
pip install -r requirements.txt
```

## Usage

### Running the Full Simulation

To run the complete simulation with all visualizations and analyses:

```python
python stock_crypto_simulation.py
```

### Running the Demo

For a quick demonstration with step-by-step explanations:

```python
python stock_crypto_demo.py
```

### Customizing the Simulation

You can modify the simulation parameters in the script to match your specific scenario:

```python
# Simulation parameters
num_trading_days = 252  # Trading days to simulate
stock_mean_return = 0.0002  # Mean daily return for stocks (~5% annually)
stock_volatility = 0.01  # Daily volatility for stocks
crypto_mean_return = 0.0008  # Mean daily return for crypto (~20% annually)
crypto_volatility = 0.04  # Daily volatility for crypto
correlation = -0.3  # Correlation between returns
```

## Mathematical Background

### Geometric Brownian Motion

The simulation uses geometric Brownian motion, which is defined by the stochastic differential equation:

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Where:
- $S_t$ is the stock price at time $t$
- $\mu$ is the drift (expected return)
- $\sigma$ is the volatility
- $W_t$ is a Wiener process (Brownian motion)

### Correlated Random Variables

To generate correlated asset returns, we use the Cholesky decomposition approach:

$$Z_2' = \rho Z_1 + \sqrt{1 - \rho^2} Z_2$$

Where:
- $Z_1$ and $Z_2$ are independent standard normal random variables
- $Z_2'$ is a standard normal random variable correlated with $Z_1$
- $\rho$ is the desired correlation coefficient

## Example Output

The simulation generates several visualizations:

1. **Price Movement Chart**: Time series plot of both asset prices
2. **Return Distribution**: Histogram showing the distribution of daily returns
3. **Statistical Summary**: Key metrics for risk and return analysis

## Limitations

- The simulation assumes constant parameters (volatility, correlation, etc.)
- It does not account for market microstructure effects or trading volume
- Real markets may exhibit fat-tailed distributions not captured by normal distributions

## Extensions

Possible extensions to this project include:

- Adding jump-diffusion processes to model market crashes
- Implementing GARCH models for time-varying volatility
- Including more asset classes for portfolio analysis
- Adding trading strategies and backtesting capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The mathematical models are based on standard financial literature
- Inspiration drawn from quantitative finance and econophysics research