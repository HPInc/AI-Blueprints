# HP Reinsurance Portfolio Analysis

## Overview
This repository contains a Jupyter notebook that demonstrates HP's Jupyter capabilities through a comprehensive reinsurance portfolio analysis. The notebook generates synthetic insurance policy data, performs statistical analysis, and creates interactive visualizations to help insurance professionals understand their portfolio distribution and risk exposure.

## Features
- **Synthetic Data Generation**: Creates realistic insurance policy data with industry-specific risk parameters and policy limits
- **Reinsurance Band Classification**: Categorizes policies into five standard reinsurance bands based on their policy limits
- **Statistical Analysis**: Provides detailed portfolio metrics including distribution across bands, industry exposure, and risk score analysis
- **Interactive Visualizations**: Generates multiple plots to visualize portfolio characteristics

## Requirements
- Python 3.x
- Required libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - datetime

## File Structure
- `hp_reinsurance_demo.ipynb`: Main Jupyter notebook containing all code and analysis

## Key Components

### Data Generation
The notebook generates synthetic insurance policy data with:
- Industry categories (Manufacturing, Healthcare, Technology, Retail, Energy)
- Risk scores tailored to each industry
- Policy limits based on industry and risk score

### Reinsurance Band Classification
Policies are categorized into five standard bands:
- Band 1: Up to $1M
- Band 2: $1M - $5M
- Band 3: $5M - $10M
- Band 4: $10M - $50M
- Band 5: $50M+

### Analysis Functions
The notebook includes functions for:
- Calculating basic portfolio statistics
- Analyzing distribution across bands
- Examining industry exposure within bands
- Visualizing relationships between risk scores and policy limits

### Visualizations
The notebook generates four main visualizations:
1. Distribution of policies across reinsurance bands
2. Risk score distribution by reinsurance band
3. Industry distribution within each band
4. Relationship between risk scores and policy limits by industry

## Usage
1. Ensure all required libraries are installed
2. Open the Jupyter notebook in your preferred environment
3. Run all cells to generate the synthetic data and analysis
4. Modify parameters as needed to simulate different portfolio scenarios

## Business Applications
This analysis can be used to:
- Optimize reinsurance strategy
- Manage exposure concentrations
- Understand industry-specific risks
- Identify potential areas for growth or reduction in the portfolio

## Notes
- The data generation uses a fixed random seed (42) for reproducibility
- All visualizations use customized styling for better readability
- The notebook is designed as a demonstration and can be adapted for real-world portfolio data