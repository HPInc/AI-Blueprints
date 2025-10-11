import os
import re
import numpy as np
import pandas as pd
import math
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing

# Initialize TF-IDF vertorizer for semantic similarity
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)


def semantic_similarity_eval_fn(predictions: List[str], targets: List[str]) -> float:
    """
    Compute semantic similarity between predictions and targets using TF-IDF and cosine similarity.

    Args:
        predictions (List[str]): Model-generated texts.
        targets (List[str]): Ground truth references.

    Returns:
        float: Mean cosine similarity between matched pairs.
    """
    all_texts = list(targets) + list(predictions)

    all_texts = [str(text) if text else "" for text in all_texts]

    if len(set(all_texts)) < 2:
        return 10.0

    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)

    n_targets = len(targets)
    target_vectors = tfidf_matrix[:n_targets]
    pred_vectors = tfidf_matrix[n_targets:]

    similarities = []
    for i in range(len(targets)):
        similarity = cosine_similarity(
            target_vectors[i : i + 1], pred_vectors[i : i + 1]
        )[0][0]
        similarities.append(similarity)

    return np.mean(similarities) * 10


def _count_syllables(word: str) -> int:
    """Rudimentary English syllable counter."""
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e"):
        count = max(1, count - 1)
    return max(count, 1)


def flesch_reading_ease(text: str) -> float:
    """Calculates Flesch Reading Ease score."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    words = re.findall(r"\w+", text)
    if not sentences or not words:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    W = len(words)
    S = len(sentences)
    score = 206.835 - 1.015 * (W / S) - 84.6 * (syllables / W)
    return score


def readability_improvement_eval_fn(
    predictions: List[str], targets: List[str]
) -> float:
    """
    Calculates the average absolute change in Flesch Reading Ease.
    A score of 0.0 means no change in readability on average.
    Positive values indicate improvement, negative values indicate a decrease.
    """
    deltas = []
    for pred, target in zip(predictions, targets):
        # Ensure inputs are strings and handle if they are empty
        orig_text = str(target)
        pred_text = str(pred)

        if not orig_text.strip():
            orig_score = 0.0
        else:
            orig_score = flesch_reading_ease(orig_text)

        if not pred_text.strip():
            new_score = 0.0
        else:
            new_score = flesch_reading_ease(pred_text)

        # Calculate the simple difference (delta)
        deltas.append(new_score - orig_score)

    # If the list is empty for any reason, return 0
    if not deltas:
        return 0.0

    # Return the average of all the deltas directly.
    return float(np.mean(deltas))


def llm_judge_eval_fn_local(predictions: pd.Series, targets: pd.Series, llm) -> float:
    """
    Use the main Llama 3 model to rate the grammar of predictions from 1 to 10.
    This function correctly handles Pandas Series as input from MLflow.
    """
    # Use the correct method to check if a Pandas Series is empty
    if predictions.empty:
        return 0.0

    scores = []
    # Iterating over a Series works as expected
    for pred in predictions:
        # This prompt is formatted for Llama 3 Instruct
        prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a grammar judge. Rate the grammar of the provided text on a scale from 1 (very poor) to 10 (perfect).
Respond with a single integer number only. Do not add any explanation or punctuation.

Keep in mind the text is markdown text, so do not penalize the use of markdown syntax such as *, #, `, [].

Example:
Text: I has a apple.
Answer: 7

Text: The dog chased **the ball** across the yard.
Answer: 10<|eot_id|><|start_header_id|>user<|end_header_id|>
Text: {pred}
Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        try:
            result = llm.invoke(prompt)
            text = result.strip()
            match = re.search(r"\d+", text)
            score = float(match.group(0)) if match else 0.0
            scores.append(score)
        except Exception as e:
            print(f"[LLM judge runtime error]: {e}")
            scores.append(5.0)

    return float(np.mean(scores)) if scores else 0.0
