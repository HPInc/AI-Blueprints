import pandas as pd
import re

from mlflow.metrics import make_metric, MetricValue
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate

class LocalGenAIJudge:
    META_LLAMA_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    SYSTEM_PROMPT = (
        "You are an evaluation model judging the quality of AI-generated answers.\n"
        "Your job is to output a single number between 0.0 and 1.0 based on the criteria given.\n\n"
        "Rules:\n"
        "- Only use the provided context.\n"
        "- Do not use any external knowledge.\n"
        "- Output only the numeric score without explanation.\n"
    )

    def __init__(self, llm: LlamaCpp):
        self.llm = llm

        self.prompt_template = PromptTemplate(
            template=self.META_LLAMA_TEMPLATE,
            input_variables=["system_prompt", "user_prompt"],
        )

    def _run_prompt(self, user_prompt: str) -> float:
        prompt = self.prompt_template.format(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        response = self.llm.invoke(prompt)
        try:
            return float(response.strip())
        except Exception:
            return 0.0

    @staticmethod
    def _extract_score(response: str) -> float:
        # find the first 0–1 float, e.g. "0.8" or "1.0"
        m = re.search(r"\b0(?:\.\d+)?|1(?:\.0+)?\b", response)
        return float(m.group()) if m else 0.0

    
    def _evaluate(self, batch_df: pd.DataFrame, prompt_builder) -> pd.Series:
        # 1) build all prompts for the batch in one go
        prompts = [
            prompt_builder(
                question=row.questions,
                answer=row.result,
                context=row.source_documents
            )
            for row in batch_df.itertuples()
        ]
    
        # 2) invoke the LlamaCpp batch API
        raw = self.llm.batch(prompts, stop=["\n"])
    
        # 3) parse floats and preserve original index ordering
        scores = [self._extract_score(r) for r in raw]
        return pd.Series(scores, index=batch_df.index)

    def evaluate_faithfulness(self, batch_df: pd.DataFrame) -> pd.Series:
        def prompt_builder(question, answer, context):
            return (
                f"<context>\n{context}\n</context>\n\n"
                f"Question: {question}\n"
                f"Answer: {answer}\n\n"
                "Rate how faithful the answer is to the context on a scale from 0.0 (not faithful) to 1.0 (perfectly faithful).\n"
                "Score:"
            )
        return self._evaluate(batch_df, prompt_builder)

    def evaluate_relevance(self, batch_df: pd.DataFrame) -> pd.Series:
        def prompt_builder(question, answer, context):
            return (
                f"Question: {question}\n"
                f"Answer: {answer}\n\n"
                "Rate how relevant the answer is to the question on a scale from 0.0 (irrelevant) to 1.0 (highly relevant).\n"
                "Score:"
            )
        return self._evaluate(batch_df, prompt_builder)

    def evaluate_correctness(self, batch_df: pd.DataFrame) -> pd.Series:
        def prompt_builder(question, answer, context):
            return (
                f"<context>\n{context}\n</context>\n\n"
                f"Question: {question}\n"
                f"Answer: {answer}\n\n"
                "Rate how factually correct the answer is based on the context on a scale from 0.0 (incorrect) to 1.0 (fully correct).\n"
                "Score:"
            )
        return self._evaluate(batch_df, prompt_builder)

    def evaluate_similarity(self, batch_df: pd.DataFrame) -> pd.Series:
        def prompt_builder(question, answer, context):
            return (
                f"Generated Answer: {answer}\n"
                f"Ground Truth: {context}\n\n"
                "Rate how semantically similar the two responses are on a scale from 0.0 (completely different) to 1.0 (identical meaning).\n"
                "Score:"
            )
        return self._evaluate(batch_df, prompt_builder)

    def to_mlflow_metric(self, metric_name: str):
        scorers = {
            "faithfulness": self.evaluate_faithfulness,
            "relevance":    self.evaluate_relevance,
            "correctness":  self.evaluate_correctness,
            "similarity":   self.evaluate_similarity,
        }
        if metric_name not in scorers:
            raise ValueError(f"Unsupported metric: {metric_name}")

        scorer = scorers[metric_name]

        def _eval_fn(
            predictions: pd.Series,
            inputs: pd.Series,
            context: pd.Series | None = None,
            targets: pd.Series | None = None,
        ):
            df = pd.DataFrame(
                {
                    "questions": inputs,
                    "result":    predictions,
                    "source_documents":
                        context if context is not None
                        else [""] * len(predictions),
                }
            )
            scores = scorer(df)                 # pd.Series of floats
            return MetricValue(scores=scores.tolist())  # <-- no aggregations

        return make_metric(
            eval_fn=_eval_fn,
            name=f"local_{metric_name}",
            greater_is_better=True,
            long_name=f"Local Llama {metric_name}",
        )