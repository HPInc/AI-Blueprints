import pandas as pd
import re
import torch
from transformers import pipeline, AutoTokenizer
from mlflow.metrics import make_metric, MetricValue

class LocalGenAIJudge:
    """
    A dual-purpose evaluation judge using a local generative AI model.
    
    1. Can be used standalone with `mlflow.evaluate` via the `to_mlflow_metric` method.
    2. Can be integrated into a model's `predict` method for real-time scoring.
    """
    SYSTEM_PROMPT = (
        "You are a meticulous and impartial expert AI judge. Your role is to evaluate an AI-generated answer based on a specific criterion and the provided context."
        "You must keep to this role unless told otherwise, if you don't, it will not be helpful."
        "Only output a single floating-point number between 0.0 and 1.0 and nothing else."
        "Do not provide any explanation, preamble, or additional text. Your entire response must be only the numeric score. Do not hallucinate."
    )

    def __init__(self, model: any, tokenizer: any):
        """
        Initializes the judge directly with a model and tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer

    def _build_prompt(self, user_prompt: str) -> str:
        """Applies the chat template to the system and user prompts."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @staticmethod
    def _extract_score(response: str) -> float:
        """Extracts the first valid score (0.0-1.0) from the model's output."""
        match = re.search(r"\b(0(\.\d+)?|1(\.0+)?)\b", response)
        if match:
            try:
                return float(match.group(0))
            except (ValueError, IndexError):
                return 0.0
        return 0.0

    def _evaluate(self, batch_df: pd.DataFrame, prompt_builder) -> pd.Series:
        """Internal method to run batch evaluation for a given metric."""
        user_prompts = [
            prompt_builder(
                question=row.questions,
                answer=row.result,
                context=row.source_documents
            )
            for row in batch_df.itertuples()
        ]
        
        full_prompts = [self._build_prompt(p) for p in user_prompts]
        
        responses_text = []
        for prompt in full_prompts:
            res = self.model.chat(
                self.tokenizer,
                None,
                prompt,
                generation_config=dict(
                    max_new_tokens=16,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            )
            # -----------------------------------
            responses_text.append(res)

        # Adjust score extraction to work with a simple list of strings
        scores = [self._extract_score(res) for res in responses_text]
        return pd.Series(scores, index=batch_df.index, dtype=float)

    def evaluate_faithfulness(self, batch_df: pd.DataFrame) -> pd.Series:
        """Scores how faithful the answer is to the provided context."""
        def prompt_builder(question, answer, context):
            return (
                "Please evaluate the factual alignment of the 'Answer' to the 'Context' by providing a continuous score from 0.0 to 1.0.\n\n"
                f"**Context:**\n---\n{context}\n---\n\n"
                f"**Question:** {question}\n\n"
                f"**Answer:** {answer}\n\n"
                "**Scoring Guidelines:**\n"
                "- A score of **1.0** represents **Perfect Alignment**, where every claim in the Answer is directly and explicitly supported by the Context.\n"
                "- A score of **0.0** represents **No Alignment**, where the Answer significantly contradicts the Context or is a complete hallucination.\n"
                "- Use **intermediate scores** (e.g., 0.25, 0.7, 0.9) to reflect the degree of factual support. A higher score indicates fewer and less severe unsupported claims.\n"
                "For example, an answer that is mostly correct but contains one minor unsupported detail might score a **0.85**.\n\n"
                "Output only the numeric score."
            )
        return self._evaluate(batch_df, prompt_builder)

    def evaluate_relevance(self, batch_df: pd.DataFrame) -> pd.Series:
        """Scores how relevant the answer is to the question."""
        def prompt_builder(question, answer, context):
            return (
                "Please evaluate how well the 'Answer' addresses the 'Question' by providing a continuous score from 0.0 to 1.0.\n\n"
                f"**Question:** {question}\n\n"
                f"**Answer:** {answer}\n\n"
                "**Scoring Guidelines:**\n"
                "- A score of **1.0** represents **Perfect Relevance**, where the Answer directly, completely, and concisely "
                "addresses all parts of the user's question and intent.\n"
                "- A score of **0.0** represents **Complete Irrelevance**, where the Answer is off-topic or fails to address the question in any meaningful way.\n"
                "- Use **intermediate scores** (e.g., 0.25, 0.7, 0.9) to reflect the degree of relevancy. A higher score indicates more relevant, complete responses.\n"
                "For example, an answer that correctly addresses the main topic but misses a secondary part of the question might score a **0.6**.\n\n"
                "Output only the numeric score."
            )
        return self._evaluate(batch_df, prompt_builder)
        
    def to_mlflow_metric(self, metric_name: str):
        """
        Factory method to create an MLflow Metric object for use with mlflow.evaluate().
        """
        scorers = {
            "faithfulness": self.evaluate_faithfulness,
            "relevance": self.evaluate_relevance,
        }
        if metric_name not in scorers:
            raise ValueError(f"Unsupported metric: {metric_name}")

        scorer_fn = scorers[metric_name]

        def _eval_fn(
            predictions: pd.Series,
            inputs: pd.Series,
            context: pd.Series
        ):
            df = pd.DataFrame({
                "questions": inputs,
                "result": predictions,
                "source_documents": context,
            })
            scores = scorer_fn(df)
            return MetricValue(scores=scores.tolist())

        return make_metric(
            eval_fn=_eval_fn,
            name=f"local_judge_{metric_name}",
            greater_is_better=True,
            long_name=f"Local Judge - {metric_name.capitalize()}",
        )