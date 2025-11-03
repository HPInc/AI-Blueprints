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


def normalize_ratings(ratings, min_rating=1, max_rating=5):
    """Normalize ratings to 1-5 scale"""
    ratings = np.array(ratings)

    if ratings.max() == ratings.min():
        return np.full(ratings.shape, 3.0)  # Return middle value

    normalized = (ratings - ratings.min()) / (ratings.max() - ratings.min())
    scaled = normalized * (max_rating - min_rating) + min_rating

    return np.clip(scaled, min_rating, max_rating)


class Model:
    """
    Standalone movie recommendation model containing all business logic.
    NO MLflow inheritance - pure domain functionality.

    Uses memory-based collaborative filtering with item similarity to generate
    personalized movie recommendations based on user input.
    """

    def __init__(
        self,
        train_data_matrix: np.ndarray,
        movie_titles: pd.DataFrame,
        config: Dict[str, Any],
    ):
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
        self.neutral_rating = 3.0
        self.personalization_weight = 2.0
        self.top_n = 5

        # Replace zeros with NaN so they don't affect the average
        data_with_nan = np.where(
            self.train_data_matrix > 0, self.train_data_matrix, np.nan
        )
        self.mean_ratings = np.nanmean(data_with_nan, axis=0)
        self.mean_ratings = np.nan_to_num(self.mean_ratings, nan=0.0)
        logger.info(
            f"✅ Model initialized with {self.n_users} users and {self.n_items} items"
        )

    def get_movie_title(self, movie_id):
        """
        Returns the movie title for a given movie_id.

        Parameters:
            movie_id (int): The ID of the movie.

        Returns:
            str: The title of the movie, or None if not found.
        """
        try:
            title_row = self.movie_titles[self.movie_titles["item_id"] == movie_id]
            if not title_row.empty:
                return title_row.iloc[0]["title"]
            else:
                logger.warning(f"⚠️ Movie ID {movie_id} not found in titles.")
                return None
        except Exception as e:
            logger.error(f"❌ Error retrieving movie title: {str(e)}")
            return None

    def predict(
        self, model_input: Dict[str, Any], params=None
    ) -> List[Tuple[str, float]]:
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
            movie_ids = model_input["movie_id"].tolist()
            ratings = model_input["rating"].tolist()

            # Calculate predictions for all movies
            predictions = np.zeros(len(self.mean_ratings))

            for movie_id, rating in zip(movie_ids, ratings):
                movie_idx = int(movie_id) - 1  # Convert to array index

                # Skip invalid movie IDs
                if movie_idx < 0 or movie_idx >= len(predictions):
                    continue

                # How much user likes/dislikes compared to neutral (3.0)
                weight = (float(rating) - self.neutral_rating) / 2.0

                item_similarity = 1 - pairwise_distances(
                    self.train_data_matrix.T, metric="cosine"
                )
                # Add similar movies weighted by user preference
                predictions += item_similarity[movie_idx] * weight

            # Combine user preference with popular movies
            final_predictions = (
                self.mean_ratings + predictions * self.personalization_weight
            )

            # Normalize to 1-5 scale
            final_predictions = normalize_ratings(final_predictions)

            # Create list of (title, score, index)
            movie_titles = self.movie_titles["title"].tolist()
            rated_indices = set(int(mid) - 1 for mid in movie_ids)

            all_movies = []
            for idx, (title, score) in enumerate(zip(movie_titles, final_predictions)):
                if idx not in rated_indices:  # Skip already rated
                    all_movies.append((title, float(score)))

            # Sort by score and return top N
            all_movies.sort(key=lambda x: x[1], reverse=True)
            return all_movies[: self.top_n]

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
            "algorithm": "item_based_collaborative_filtering",
        }
