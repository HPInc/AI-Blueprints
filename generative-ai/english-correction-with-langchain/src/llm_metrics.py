import os
import re
import numpy as np
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mlflow.metrics import make_metric
from llama_cpp import Llama

# Initialize TF-IDF vertorizer for semantic similarity
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Path to local judge model
LOCAL_LLAMA_JUDGE_PATH = "/home/jovyan/datafabric/llama2-7b/ggml-model-f16-Q5_K_M.gguf" 

class LocalJudgeLlamaClient:
    """Singleton wrapper for local judge-specific LLaMA."""
    _client = None

    @classmethod
    def get_client(cls, model_path: Optional[str] = None) -> Llama:
        """
        Get or initialize the singleton LLaMA client.

        Args:
            model_path (Optional[str]): Path to the local gguf LLaMA model.

        Returns:
            Llama: Loaded LLaMA instance.
        """
        if cls._client is None:
            if model_path is None:
                raise ValueError("Must provide model_path to initialize local LLaMA judge.")

            cls._client = Llama(
                model_path=model_path,
                n_ctx=512,           
                n_gpu_layers=-1,
                n_batch=8,
                f16_kv=True,
                temperature=0.0,     
                max_tokens=32,
                stop=["\n"]
            )
        return cls._client

# Preload model at module load
LocalJudgeLlamaClient.get_client(model_path=LOCAL_LLAMA_JUDGE_PATH)

def simple_grammar_check(text: str) -> int:
    """
    Basic grammar checking without external libraries.
    Detects repeated words, double spaces, and capitalization issues.

    Args:
        text (str): Input sentence to analyze.

    Returns:
        int: Count of potential grammar issues.
    """
    issues = 0
    text = str(text).strip()
    
    # Check for basic issues
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check for sentences not starting with capital letter
        if sentence and not sentence[0].isupper():
            issues += 1
            
        # Check for double spaces
        if '  ' in sentence:
            issues += 1
            
        # Check for common grammar patterns
        words = sentence.lower().split()
        for i, word in enumerate(words):
            # Basic subject-verb agreement checks
            if word == 'i' and i < len(words) - 1:
                if words[i + 1] in ['are', 'were']:
                    issues += 1  
                    
            # Check for repeated words
            if i > 0 and word == words[i-1]:
                issues += 1
                
    return issues

def semantic_similarity_eval_fn(predictions: List[str], targets: List[str]) -> float:
    """
    Compute semantic similarity between predictions and targets using TF-IDF and cosine similarity.

    Args:
        predictions (List[str]): Model-generated texts.
        targets (List[str]): Ground truth references.

    Returns:
        float: Mean cosine similarity between matched pairs.
    """
    # Combine all texts to fit the vectorizer
    all_texts = list(targets) + list(predictions)
    
    # Handle empty texts
    all_texts = [str(text) if text else "" for text in all_texts]
    
    if len(set(all_texts)) < 2:  # All texts are identical or empty
        return 1.0
    
    # Fit and transform
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
    
    # Split back into targets and predictions
    n_targets = len(targets)
    target_vectors = tfidf_matrix[:n_targets]
    pred_vectors = tfidf_matrix[n_targets:]
    
    # Calculate cosine similarity for each pair
    similarities = []
    for i in range(len(targets)):
        similarity = cosine_similarity(target_vectors[i:i+1], pred_vectors[i:i+1])[0][0]
        similarities.append(similarity)
    
    return np.mean(similarities)

def grammar_error_count_eval_fn(predictions, targets):
    """
    Count grammar issues in the predictions.

    Args:
        predictions (List[str]): Model outputs.
        targets (List[str]): Reference texts (unused here).

    Returns:
        float: Average number of issues per prediction.
    """
    error_counts = []
    for pred in predictions:
        error_count = simple_grammar_check(str(pred))
        error_counts.append(error_count)
    
    return np.mean(error_counts)

def grammar_error_rate_eval_fn(predictions: List[str], targets: List[str]) -> float:
    """
    Calculate grammar error rate (issues per word) for predictions.

    Args:
        predictions (List[str]): Model outputs.
        targets (List[str]): Reference texts (unused here).

    Returns:
        float: Mean error rate.
    """
    error_rates = []
    for pred in predictions:
        error_count = simple_grammar_check(str(pred))
        word_count = len(str(pred).split())
        error_rate = error_count / max(word_count, 1)  # Avoid division by zero
        error_rates.append(error_rate)
    
    return np.mean(error_rates)

def grammar_improvement_eval_fn(predictions: List[str], targets: List[str]) -> float:
    """
    Measure improvement in grammar (fewer errors) from targets to predictions.

    Args:
        predictions (List[str]): Corrected text.
        targets (List[str]): Original input text.

    Returns:
        float: Mean improvement (positive = fewer errors in prediction).
    """
    improvements = []
    for pred, target in zip(predictions, targets):
        input_errors = simple_grammar_check(str(target))
        output_errors = simple_grammar_check(str(pred))
        improvement = input_errors - output_errors  
        improvements.append(improvement)
    
    return np.mean(improvements)

def grammar_score_eval_fn(predictions: List[str], targets: List[str]) -> float:
    """
    Assign grammar score from 0–100 based on number of issues.

    Args:
        predictions (List[str]): Model outputs.
        targets (List[str]): Reference texts (unused here).

    Returns:
        float: Mean score where 100 = perfect grammar.
    """
    scores = []
    for pred in predictions:
        error_count = simple_grammar_check(str(pred))
        word_count = len(str(pred).split())
        if word_count == 0:
            scores.append(0)
        else:
            # Simple scoring: start at 100, subtract points for errors
            error_penalty = min(error_count * 10, 100)  
            score = max(100 - error_penalty, 0)
            scores.append(score)
    
    return np.mean(scores)

def readability_improvement_eval_fn(predictions: List[str], targets: List[str]) -> float:
    """
    Estimate improvement in readability using sentence length as proxy.

    Args:
        predictions (List[str]): Corrected output.
        targets (List[str]): Original input.

    Returns:
        float: Average improvement in readability score.
    """
    def calculate_readability_score(text: str) -> float:
        """
        Estimate a basic readability score for a given text.
    
        Uses average sentence length as a simple heuristic. Shorter sentences are considered more readable.
    
        Args:
            text (str): The input text to evaluate.
    
        Returns:
            float: Readability score where higher is better.
        """
        sentences = text.split('.')
        words = text.split()
        if len(sentences) == 0 or len(words) == 0:
            return 0
        
        avg_sentence_length = len(words) / len(sentences)
        # Simple readability: prefer shorter sentences and common words
        readability = max(20 - avg_sentence_length, 0)  
        return readability
    
    improvements = []
    for pred, target in zip(predictions, targets):
        input_readability = calculate_readability_score(str(target))
        output_readability = calculate_readability_score(str(pred))
        improvement = output_readability - input_readability
        improvements.append(improvement)
    
    return np.mean(improvements)

def llm_judge_eval_fn_local(predictions: List[str]) -> float:
    """
    Use a local LLaMA model to rate grammar of predictions from 1 to 10.

    Args:
        predictions (List[str]): Model outputs.

    Returns:
        float: Average LLaMA-generated grammar rating.
    """
    llama = LocalJudgeLlamaClient.get_client()
    scores = []

    for pred in predictions:
        prompt = f"""Rate the following text solely on grammar. Respond with a single digit from 1 to 10. DO NOT include any explanation, label, or punctuation. Reply with just the number.

Text: I has a apple.
Answer: 3

Text: The dog chased the ball across the yard.
Answer: 9

Text: Him don't know where she is.
Answer: 2

Text: {pred}
Answer:"""

        try:
            result = llama(prompt, stop=["\n"])
            text = result["choices"][0]["text"].strip()

            try:
                import os
                os.makedirs("llm_eval_logs", exist_ok=True)
                with open("llm_eval_logs/local_llama_responses.txt", 'a', encoding='utf-8') as f:
                    f.write(f"Response: {text}\n")
            except:
                pass

            score = float(re.findall(r"\d+", text)[0])
            scores.append(score)
        except Exception as e:
            print(f"[LLaMA judge error]: {e}")
            scores.append(5.0)

    return sum(scores) / len(scores)

# ---- Create all the metric wrappers for MLflow ----
    
semantic_similarity_metric = make_metric(
    eval_fn=semantic_similarity_eval_fn,
    greater_is_better=True,
    name="semantic_similarity"
)

grammar_error_count_metric = make_metric(
    eval_fn=grammar_error_count_eval_fn,
    greater_is_better=False,
    name="grammar_error_count"
)

grammar_error_rate_metric = make_metric(
    eval_fn=grammar_error_rate_eval_fn,
    greater_is_better=False,
    name="grammar_error_rate"
)

grammar_improvement_metric = make_metric(
    eval_fn=grammar_improvement_eval_fn,
    greater_is_better=True,
    name="grammar_improvement"
)

grammar_score_metric = make_metric(
    eval_fn=grammar_score_eval_fn,
    greater_is_better=True,
    name="grammar_score"
)

readability_improvement_metric = make_metric(
    eval_fn=readability_improvement_eval_fn,
    greater_is_better=True,
    name="readability_improvement"
)

llm_judge_metric_local = make_metric(
    eval_fn=llm_judge_eval_fn_local,
    greater_is_better=True,
    name="llm_judge_local_score"
)