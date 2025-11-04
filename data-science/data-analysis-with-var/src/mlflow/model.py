"""
Standalone Model class for COVID Movement Patterns with VAR (Vector Autoregression).

Business Logic Layer
- Handles Vector Autoregression forecasting for COVID-19 movement patterns
- Manages model initialization, data preprocessing, and prediction logic
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

import os
import pickle
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# Set up logger
logger = logging.getLogger(__name__)


class Model:
    """
    Standalone model class with no MLflow inheritance.
    Handles Vector Autoregression forecasting for COVID-19 movement patterns.
    """

    def __init__(
        self,
        config: dict,
        docs_path: Optional[str] = None,
        secrets: Optional[dict] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the Model with configuration and load artifacts from docs_path.

        Args:
            config: Model configuration dictionary
            docs_path: Path to documents directory containing pickle files
            secrets: Secrets dictionary (optional, for compatibility)
            model_path: Model path (optional, for compatibility)
        """
        self.config = config

        if docs_path is None:
            raise ValueError("docs_path is required to load VAR model artifacts")

        # Load all the model artifacts from docs_path
        artifacts = {}
        artifact_files = [
            "ny_model.pkl",
            "ldn_model.pkl",
            "ny_last_values.pkl",
            "ldn_last_values.pkl",
            "ny_last_raw_value.pkl",
            "ldn_last_raw_value.pkl",
            "features.pkl",
        ]

        for artifact_file in artifact_files:
            artifact_path = os.path.join(docs_path, artifact_file)
            if not os.path.exists(artifact_path):
                raise FileNotFoundError(f"Required artifact not found: {artifact_path}")

            with open(artifact_path, "rb") as f:
                artifact_name = artifact_file.replace(".pkl", "")
                artifacts[artifact_name] = pickle.load(f)

            logger.info(f"Loaded artifact: {artifact_file}")

        # Load the trained models from artifacts
        self.ny_model = artifacts["ny_model"]
        self.ldn_model = artifacts["ldn_model"]

        # Load the last values for forecasting and reverse transformation
        self.ny_last_values = artifacts["ny_last_values"]
        self.ldn_last_values = artifacts["ldn_last_values"]

        # Load the lag orders for each model
        self.ny_lag_order = self.ny_model.k_ar
        self.ldn_lag_order = self.ldn_model.k_ar

        # Load the last raw values needed for transforming differenced forecasts back
        self.ny_last_raw_value = artifacts["ny_last_raw_value"]
        self.ldn_last_raw_value = artifacts["ldn_last_raw_value"]

        # Load feature names
        self.features = artifacts["features"]

        logger.info("Model initialized successfully with all artifacts loaded")

    def check_stationarity(self, series: pd.Series) -> bool:
        """
        Check if a time series is stationary using Augmented Dickey-Fuller test.

        Args:
            series: Time series data to check

        Returns:
            bool: True if series is stationary, False otherwise
        """
        result = adfuller(series, autolag="AIC")
        return result[1] <= 0.05

    def difference_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply differencing to make time series stationary.

        Args:
            data: Time series data to difference

        Returns:
            pd.DataFrame: Differenced time series data
        """
        return data.diff().dropna()

    def rolling_back_transformation(
        self, last_raw_value: Any, forecast_output: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert differenced forecasts back to original scale.

        Args:
            last_raw_value: The last known raw values before differencing
            forecast_output: Forecasted differenced values

        Returns:
            pd.DataFrame: Forecasts transformed back to original scale
        """
        forecast_final = forecast_output.copy()

        for i, col in enumerate(self.features):
            col_forecast = f"{col}_forecast"
            # Cumulatively add differences starting from the last known value
            forecast_final[col_forecast] = (
                last_raw_value[i] + forecast_output[col_forecast].cumsum()
            )

        return forecast_final

    def predict(
        self, model_input: Dict[str, Any], params: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Computes the predicted forecast for COVID movement patterns.

        Args:
            model_input: Dictionary containing prediction parameters
                - city: City to forecast for ("New York" or "London")
                - steps: Number of forecast steps
                - new_data: Optional new data to update the model with
            params: Optional parameters dictionary (unused)

        Returns:
            pd.DataFrame: Forecasted values with date index
        """
        try:
            city = model_input.get("city", ["New York"])[0]
            steps = int(model_input.get("steps", [7])[0])
            new_data = model_input.get("new_data", None)

            if city == "New York":
                model = self.ny_model
                forecast_input = self.ny_last_values
                last_raw_value = self.ny_last_raw_value
                lag_order = self.ny_lag_order
            else:  # London
                model = self.ldn_model
                forecast_input = self.ldn_last_values
                last_raw_value = self.ldn_last_raw_value
                lag_order = self.ldn_lag_order

            # If new data is provided, update the forecast input
            if new_data is not None:
                new_df = pd.DataFrame(new_data)

                # Check if data needs differencing
                stationary = all(
                    self.check_stationarity(new_df[col]) for col in new_df.columns
                )

                if not stationary:
                    new_df = self.difference_data(new_df)

                forecast_input = new_df.values[-lag_order:]
                last_raw_value = new_data.iloc[-1].values

            # Make forecast
            fc = model.forecast(y=forecast_input, steps=steps)

            # Create DataFrame with forecasted values
            dates = pd.date_range(start=pd.Timestamp.today(), periods=steps)
            forecast_output = pd.DataFrame(
                fc, index=dates, columns=[f"{col}_forecast" for col in self.features]
            )

            # Transform differenced forecasts back to original scale
            forecast_final = self.rolling_back_transformation(
                last_raw_value, forecast_output
            )

            logger.info(
                f"Successfully generated forecast for {city} with {steps} steps"
            )
            return forecast_final

        except Exception as e:
            logger.error(f"Error performing prediction: {str(e)}")
            raise
