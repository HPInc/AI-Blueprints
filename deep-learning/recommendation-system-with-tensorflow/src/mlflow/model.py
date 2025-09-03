"""
Movie Recommendation Model - Pure Business Logic

This module contains the core movie recommendation functionality extracted from the original
MLflow service. It implements memory-based collaborative filtering using item similarity
to generate movie recommendations for users.

No MLflow dependencies - pure domain functionality for better testability and maintainability.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Model:
    """
    Standalone movie recommendation model containing all business logic.
    NO MLflow inheritance - pure domain functionality.
    
    Uses memory-based collaborative filtering with item similarity to generate
    personalized movie recommendations based on user input.
    """

    def __init__(self, train_data_matrix: np.ndarray, movie_titles: pd.DataFrame, config: Dict[str, Any]):
        """
        Direct dependency injection - no MLflow context.
        Initialize the recommendation model with training data and configuration.

        Args:
            train_data_matrix: User-item interaction matrix for training
            movie_titles: DataFrame containing movie titles and metadata
            config: Configuration dictionary with model parameters
        """
        self.train_data_matrix = train_data_matrix
        self.movie_titles = movie_titles
        self.config = config
        self.n_users, self.n_items = self.train_data_matrix.shape
        
        logger.info(f"✅ Model initialized with {self.n_users} users and {self.n_items} items")

    def normalize_ratings(self, ratings: np.ndarray, min_rating: int = 1, max_rating: int = 5) -> np.ndarray:
        """
        Normalizes rating values to specified range.

        Args:
            ratings: Array of rating values to normalize
            min_rating: Minimum rating value. Defaults to 1.
            max_rating: Maximum rating value. Defaults to 5.

        Returns:
            Normalized rating values clipped to the specified range.
        """
        ratings = np.array(ratings)
        if ratings.max() == ratings.min():
            return np.full(ratings.shape, (max_rating + min_rating) / 2)
        
        normalized_ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min()) * (max_rating - min_rating) + min_rating
        return np.clip(normalized_ratings, min_rating, max_rating)

    def predict(self, model_input: Dict[str, Any], params=None) -> List[Tuple[str, float]]:
        """
        Core business logic for generating movie recommendations.
        Extracted from original service predict method.
        
        Uses item-based collaborative filtering to recommend movies based on user input.

        Args:
            model_input: Dictionary containing 'movie_id' and 'rating' for user preference
            params: Optional parameters (for API compatibility)

        Returns:
            List of tuples containing (movie_title, predicted_rating) for top 5 recommendations.
        """
        try:
            # Extract user input
            movie_id = int(model_input['movie_id'][0])
            rating = float(model_input['rating'][0])
            
            logger.info(f"Generating recommendations for movie_id: {movie_id}, rating: {rating}")

            # Create user preference vector
            user_ratings = np.zeros(self.n_items)
            user_ratings[movie_id - 1] = rating

            # Add new user to training matrix
            ratings = np.vstack([self.train_data_matrix, user_ratings])

            # Calculate item similarity matrix
            item_similarity = pairwise_distances(ratings.T, metric='cosine')

            # Generate predictions using collaborative filtering
            mean_item_rating = ratings[:-1].mean(axis=0)  
            ratings_diff = (ratings[:-1] - mean_item_rating)
            pred = mean_item_rating + item_similarity.dot(ratings_diff.T).T[-1]

            # Normalize predictions to rating scale
            user_pred_normalized = self.normalize_ratings(pred)

            # Get movie titles for recommendations
            movie_titles = self.movie_titles['title'].tolist()

            # Combine predictions with movie titles
            predictions_with_titles = list(zip(movie_titles, user_pred_normalized))

            # Sort by predicted rating (descending)
            ordered_list = sorted(predictions_with_titles, key=lambda x: x[1], reverse=True)

            # Return top 5 recommendations
            best_five_movies = ordered_list[:5]
            
            logger.info(f"✅ Generated {len(best_five_movies)} recommendations")
            return best_five_movies
        
        except Exception as e:
            logger.error(f"❌ Error performing prediction: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model metadata and configuration.
        """
        return {
            "model_type": "MovieRecommendationModel",
            "n_users": self.n_users,
            "n_items": self.n_items,
            "config": self.config,
            "algorithm": "item_based_collaborative_filtering"
        }