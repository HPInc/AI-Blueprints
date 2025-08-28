"""
Iris Flower Classification Model - Pure Business Logic Implementation.
This module contains the core business logic for iris flower classification
using SVM and Linear Discriminant Analysis, extracted from the legacy MLflow service.
"""
import logging
from typing import Optional, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set up logger
logger = logging.getLogger(__name__)


class Model:
    """
    Standalone model class containing all iris flower classification business logic.
    NO MLflow inheritance - pure domain functionality.
    """

    def __init__(self, config: dict, **kwargs):
        """
        Direct dependency injection - no MLflow context.
        Extract all initialization logic from original service.
        
        Args:
            config: Configuration dictionary containing dataset URL and model parameters
            **kwargs: Additional initialization parameters
        """
        try:
            self.config = config
            self.scaler = None
            self.model = None
            self.acc_test = None
            
            # Extract dataset URL from config or use default
            self.dataset_url = config.get("dataset_url", "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
            
            # Initialize the model with training
            self._initialize_and_train()
            
            logger.info(f"Model initialized successfully with test accuracy: {self.acc_test}")
            
        except Exception as e:
            logger.error(f"Error during model initialization: {str(e)}")
            raise

    def _initialize_and_train(self):
        """
        Internal method to load data, train model, and evaluate.
        Contains all business logic from original load_context method.
        """
        try:
            # Load dataset (business logic from original service)
            col_name = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
            dataset = pd.read_csv(self.dataset_url, names=col_name)
            
            # Data preprocessing
            x = dataset.drop(['class'], axis=1)
            y = dataset['class']
            
            # Train-test split
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.20, random_state=1
            )
            
            # Feature scaling
            self.scaler = StandardScaler()
            x_train_scaled = self.scaler.fit_transform(x_train)
            x_test_scaled = self.scaler.transform(x_test)
            
            # Model training - First train SVM, then LDA (following original logic)
            # Note: The original code overwrites the SVM model with LDA, so we keep the final LDA model
            svm_model = SVC(kernel="rbf", gamma="scale", C=6.812920690579608)
            svm_model.fit(x_train_scaled, y_train)
            
            # Final model is LDA (as per original logic)
            self.model = LinearDiscriminantAnalysis(solver="svd")
            self.model.fit(x_train_scaled, y_train)
            
            # Calculate test accuracy
            self.acc_test = accuracy_score(y_test, self.model.predict(x_test_scaled))
            
            logger.info(f"Model training completed. Test accuracy: {self.acc_test}")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

    def predict(self, model_input: pd.DataFrame, params: Optional[dict] = None) -> List[str]:
        """
        Core business logic extracted from original service predict method.
        Remove context parameter - use instance variables instead.
        Must return same structure as original (list of strings).
        
        Args:
            model_input: DataFrame with iris flower measurements
            params: Optional parameters (maintained for API compatibility)
            
        Returns:
            List of predicted class names (strings)
        """
        try:
            # Validate input
            if not isinstance(model_input, pd.DataFrame):
                raise ValueError("model_input must be a pandas DataFrame")
                
            if self.model is None or self.scaler is None:
                raise RuntimeError("Model not properly initialized")
            
            # Feature scaling using trained scaler
            x_scaled = self.scaler.transform(model_input)
            
            # Make prediction
            prediction = self.model.predict(x_scaled)
            
            # Return as list of strings (matching original API)
            result = prediction.tolist()
            
            logger.info(f"Prediction completed for {len(model_input)} samples")
            return result
            
        except Exception as e:
            logger.error(f"Error performing prediction: {str(e)}")
            raise

    def get_model_info(self) -> dict:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_type": "LinearDiscriminantAnalysis",
            "test_accuracy": self.acc_test,
            "solver": "svd",
            "dataset_url": self.dataset_url,
            "features": ["sepal-length", "sepal-width", "petal-length", "petal-width"],
            "classes": ["setosa", "versicolor", "virginica"]
        }