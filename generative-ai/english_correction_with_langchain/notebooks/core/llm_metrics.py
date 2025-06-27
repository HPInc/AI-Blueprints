import openai
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from mlflow.metrics import make_metric

# Initialize tools (do this once at startup)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)


class OpenAIClient:
    """Singleton-like class to manage OpenAI client initialization"""
    _client = None
    
    @classmethod
    def get_client(cls, api_key=None):
        if cls._client is None:
            if api_key:
                cls._client = openai.OpenAI(api_key=api_key)
            else:
                cls._client = openai.OpenAI()  # Uses environment variable
        return cls._client


# Simple grammar checking without external dependencies
def simple_grammar_check(text):
    """
    Basic grammar checks without external libraries.
    Returns a count of potential issues.
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
                    issues += 1  # "I are" or "I were"
                    
            # Check for repeated words
            if i > 0 and word == words[i-1]:
                issues += 1
                
    return issues

def semantic_similarity_eval_fn(predictions, targets):
    """
    Calculate semantic similarity between input (targets) and output (predictions).
    Uses TF-IDF vectors instead of transformer models for compatibility.
    Higher score means meaning is better preserved.
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
    Count grammar errors in the output text using simple grammar checking.
    Lower is better (fewer errors).
    """
    error_counts = []
    for pred in predictions:
        error_count = simple_grammar_check(str(pred))
        error_counts.append(error_count)
    
    return np.mean(error_counts)

def grammar_error_rate_eval_fn(predictions, targets):
    """
    Calculate grammar error rate (errors per word) in output text.
    Lower is better.
    """
    error_rates = []
    for pred in predictions:
        error_count = simple_grammar_check(str(pred))
        word_count = len(str(pred).split())
        error_rate = error_count / max(word_count, 1)  # Avoid division by zero
        error_rates.append(error_rate)
    
    return np.mean(error_rates)

def grammar_improvement_eval_fn(predictions, targets):
    """
    Calculate improvement in grammar errors from input to output.
    Positive values mean improvement (fewer errors in output).
    """
    improvements = []
    for pred, target in zip(predictions, targets):
        input_errors = simple_grammar_check(str(target))
        output_errors = simple_grammar_check(str(pred))
        improvement = input_errors - output_errors  # Positive = improvement
        improvements.append(improvement)
    
    return np.mean(improvements)

def grammar_score_eval_fn(predictions, targets):
    """
    Calculate a grammar score (0-100) where 100 is perfect grammar.
    Higher is better.
    """
    scores = []
    for pred in predictions:
        error_count = simple_grammar_check(str(pred))
        word_count = len(str(pred).split())
        if word_count == 0:
            scores.append(0)
        else:
            # Simple scoring: start at 100, subtract points for errors
            error_penalty = min(error_count * 10, 100)  # Max penalty is 100
            score = max(100 - error_penalty, 0)
            scores.append(score)
    
    return np.mean(scores)

def readability_improvement_eval_fn(predictions, targets):
    """
    Calculate improvement in readability from input to output.
    Uses sentence length and word complexity as proxies.
    """
    def calculate_readability_score(text):
        sentences = text.split('.')
        words = text.split()
        if len(sentences) == 0 or len(words) == 0:
            return 0
        
        avg_sentence_length = len(words) / len(sentences)
        # Simple readability: prefer shorter sentences and common words
        readability = max(20 - avg_sentence_length, 0)  # Penalize long sentences
        return readability
    
    improvements = []
    for pred, target in zip(predictions, targets):
        input_readability = calculate_readability_score(str(target))
        output_readability = calculate_readability_score(str(pred))
        improvement = output_readability - input_readability
        improvements.append(improvement)
    
    return np.mean(improvements)


def llm_judge_eval_fn(predictions, targets):
    """
    Use GPT to judge the quality of grammar corrections.
    Returns a score from 0-10 where 10 is perfect correction.
    """
    client = OpenAIClient.get_client()
    scores = []
    
    for pred, target in zip(predictions, targets):
        prompt = f"""Rate the quality of this grammar correction on a scale of 0-10, where:
- 10: Perfect correction that fixes all errors while preserving meaning and style
- 7-9: Good correction with minor issues
- 4-6: Adequate correction but with some problems
- 1-3: Poor correction with major issues
- 0: Made the text worse or changed meaning incorrectly

Original text: "{target}"
Corrected text: "{pred}"

Consider:
1. Were grammar errors actually fixed?
2. Was meaning preserved?
3. Does it sound natural?
4. Were unnecessary changes avoided?

Reply with just the number (0-10):"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Cheaper and faster than gpt-4
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0
            )
            
            # Extract score from response
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            scores.append(score)
            
        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            scores.append(5.0)  # Default score on error
    
    return sum(scores) / len(scores)  # Return average score


def generate_gpt_gold_standards(original_texts, api_key=None):
    """Generate gold standard corrections using GPT"""
    client = OpenAIClient.get_client(api_key)
    gold_standards = []
    
    for original in original_texts:
        prompt = f"""Fix only grammatical errors in this text. Preserve all formatting exactly. Do not include any additional notes or comments. 

IMPORTANT: Text contains PLACEHOLDER tokens (like __PLACEHOLDER_1__) that represent protected content. Leave ALL placeholders exactly as they are. They must all be present in the output.

Text to correct:
{original}

Corrected text:"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            corrected = response.choices[0].message.content.strip()
            gold_standards.append(corrected)
            
        except Exception as e:
            print(f"Error generating gold standard: {e}")
            gold_standards.append(original)  # Fallback
    
    return gold_standards


# Create all the metric instances
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

# Create the metric instance
llm_judge_metric = make_metric(
    eval_fn=llm_judge_eval_fn,
    greater_is_better=True,
    name="llm_judge_score"
)