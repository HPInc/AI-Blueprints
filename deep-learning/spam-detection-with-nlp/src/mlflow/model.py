"""
Standalone Spam Detection Model class.

Business Logic Layer
- Handles text preprocessing and spam classification using NLP techniques
- Manages NLTK dependencies, data loading, and machine learning pipeline
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

import os
import logging
import string
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Configure logger
logger = logging.getLogger(__name__)


class Model:
    """
    Standalone spam detection model class containing all business logic.
    NO MLflow inheritance - pure domain functionality.
    """

    def __init__(self, data_path: str, nltk_data_path: str, config: Dict[str, Any]):
        """
        Direct dependency injection - no MLflow context.
        Extract all initialization logic from original service.

        Args:
            data_path: Path to the spam dataset CSV file
            nltk_data_path: Path to NLTK data directory
            config: Configuration dictionary
        """
        self.data_path = data_path
        self.nltk_data_path = nltk_data_path
        self.config = config
        self.pipeline = None
        self.stop_words = None
        self._X_test = None
        self._y_test = None

        # Initialize the model
        self._setup_nltk()
        self._train_model()

    def _setup_nltk(self):
        """
        Set up NLTK data path and download stopwords if needed.
        """
        try:
            # Ensure NLTK data directory exists and is properly configured
            nltk_dir = os.path.abspath(self.nltk_data_path)

            # Add our custom path to the beginning of NLTK data paths
            if nltk_dir not in nltk.data.path:
                nltk.data.path.insert(0, nltk_dir)
            os.environ["NLTK_DATA"] = nltk_dir

            # Ensure stopwords are available
            self._ensure_local_stopwords(nltk_dir)

            # Try to load stopwords, with fallback to default NLTK path
            try:
                self.stop_words = set(stopwords.words("english"))
            except LookupError:
                logger.warning(
                    "Stopwords not found in custom path, trying to download to default location"
                )
                nltk.download("stopwords", quiet=True)
                self.stop_words = set(stopwords.words("english"))

        except Exception as e:
            logger.error(f"Error setting up NLTK: {str(e)}")
            raise

    def _ensure_local_stopwords(self, base_dir: str):
        """
        Ensure stopwords are available locally.

        Args:
            base_dir: Base directory for NLTK data
        """
        sw_file = Path(base_dir) / "corpora" / "stopwords" / "english"
        if not sw_file.exists():
            sw_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info("⬇️ Downloading stopwords to %s ...", base_dir)
            try:
                nltk.download(
                    "stopwords", download_dir=base_dir, quiet=True, raise_on_error=True
                )
            except Exception as e:
                logger.warning(f"Failed to download stopwords to custom path: {e}")
                # Fallback: try downloading to default NLTK path
                logger.info("Trying to download stopwords to default NLTK path...")
                nltk.download("stopwords", quiet=True)

    def preprocess(self, text: str) -> List[str]:
        """
        Preprocesses the message, performing:
        1. Removal of all punctuation
        2. Removal of all stopwords
        3. Return of a list of the cleaned text

        Args:
            text: Input text to preprocess

        Returns:
            List of cleaned tokens
        """
        try:
            text = text.lower()
            nopunc = "".join(c for c in text if c not in string.punctuation)
            return [w for w in nopunc.split() if w and w not in self.stop_words]

        except Exception as e:
            logger.error(f"Error preprocessing: {str(e)}")
            raise

    def _train_model(self):
        """
        Load data and train the spam detection pipeline.
        """
        try:
            # Load and clean the data
            df = pd.read_csv(
                self.data_path, sep=",", names=["label", "message", "v3", "v4", "v5"]
            )

            # Clean the data
            df = df.dropna(subset=["label", "message"])

            # Balance the dataset by downsampling 'ham' to match 'spam'
            ham_msg = df[df["label"] == "ham"]
            spam_msg = df[df["label"] == "spam"]

            ham_msg_balanced = ham_msg.sample(n=len(spam_msg), random_state=42)

            # Combine the balanced data
            df = pd.concat([ham_msg_balanced, spam_msg]).reset_index(drop=True)

            # Split the data
            X_tr, X_te, y_tr, y_te = train_test_split(
                df["message"], df["label"], test_size=0.2, random_state=42
            )

            # Create and train the pipeline
            self.pipeline = Pipeline(
                [
                    ("bow", CountVectorizer(analyzer=self.preprocess)),
                    ("tfidf", TfidfTransformer()),
                    ("clf", MultinomialNB(alpha=1.0)),
                ]
            )

            self.pipeline.fit(X_tr, y_tr)
            self._X_test, self._y_test = X_te, y_te

            logger.info("Model training completed successfully")

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict(
        self,
        model_input: Union[pd.DataFrame, List[str], str],
        params: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Core business logic extracted from original service predict method.
        Remove context parameter - use instance variables instead.
        Must return same pandas.DataFrame structure as original.

        Args:
            model_input: Input text(s) for prediction - can be DataFrame, list, or string
            params: Optional parameters (unused but kept for compatibility)

        Returns:
            List of predictions ('ham' or 'spam')
        """
        try:
            if self.pipeline is None:
                raise ValueError("Model not trained. Call _train_model() first.")

            # Handle different input types
            if hasattr(model_input, "values"):
                # DataFrame input
                texts = model_input.values.flatten()
            elif isinstance(model_input, list):
                texts = model_input
            else:
                texts = [str(model_input)]

            # Make predictions
            predictions = self.pipeline.predict(texts)
            return predictions.tolist()

        except Exception as e:
            logger.error(f"Error performing prediction: {str(e)}")
            raise

    def get_test_accuracy(self) -> Dict[str, Any]:
        """
        Calculate accuracy on test set.

        Returns:
            Dictionary containing accuracy metrics
        """
        if self.pipeline is None or self._X_test is None:
            raise ValueError("Model not trained or test data not available")

        preds = self.pipeline.predict(self._X_test)
        report = classification_report(self._y_test, preds, output_dict=True)

        return {"accuracy": report["accuracy"], "classification_report": report}
