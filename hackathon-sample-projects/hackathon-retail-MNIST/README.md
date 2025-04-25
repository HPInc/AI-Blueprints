# Fashion MNIST: Simple vs. Deeper CNN Comparison

## Project Overview
This project compares two convolutional neural network architectures on the Fashion MNIST dataset:
- **SimpleCNN**: A basic CNN with 2 convolutional layers.
- **DeeperCNN**: A deeper CNN with 4 convolutional layers.

The goal is to evaluate and visualize the marginal performance differences between these models using accuracy, loss, ROC curves, and probability-based visualizations.

## Directory Structure
```
FASHJION/
├── configs/
│   └── config.yaml
├── docs/
│   ├── best_practices.txt
│   └── task.txt
├── data/           # Downloaded FashionMNIST data
├── fashion_mnist_comparison.ipynb
├── requirements.txt
└── README.md
```

## How to Run
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the notebook:**
   Open `fashion_mnist_comparison.ipynb` in Jupyter Lab and execute all cells sequentially.

## Configuration
- All experiment parameters are stored in `configs/config.yaml` (no hardcoded values in code).
- You can adjust batch size, learning rate, epochs, etc. in this YAML file.

## Results & Visualizations
- **Accuracy & Loss**: Bar plots compare both models on the test set.
- **ROC Curves**: Macro-averaged ROC curves for both models.
- **Probability Difference Histogram**: Highlights marginal differences in model confidence.
- **Scatter Plot**: Per-sample probability comparison for the true class.
- **Sample Visualization**: Random FashionMNIST images with class labels.

## Minimum Hardware Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 1GB free disk space
- **GPU**: Optional (for faster training)

## Reproducibility
- All random seeds are set in the config file.
- Results are reproducible if run with the same config and environment.

## Best Practices Followed
- Explicit configuration and dependency management
- Logging instead of print statements
- Type hints and docstrings for all functions
- No hardcoded values in code
- Clear project structure and documentation
- Visual and statistical model comparison

## References
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- See `docs/best_practices.txt` for full coding and repo guidelines.
