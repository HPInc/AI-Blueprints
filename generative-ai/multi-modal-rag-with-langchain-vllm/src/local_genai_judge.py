import pandas as pd
import re
from vllm import SamplingParams

class LocalGenAIJudge:
    """
    A dual-purpose evaluation judge using a local generative AI model served by vLLM.
    
    1. Can be used standalone with `mlflow.evaluate`.
    2. Can be integrated into a model's `predict` method for real-time scoring.
    """
    SYSTEM_PROMPT = (
        "You are a meticulous AI Judge. Your role is to assess an AI's response against "
        "specific criteria based on provided materials. "
        "Internally, reason step-by-step to determine your score. "
        "Your final output, however, MUST be a single floating-point number between 0.0 and 1.0 "
        "and absolutely nothing else. Adhere strictly to this format."
    )

    def __init__(self, llm: any, tokenizer: any):
        """
        Initializes the judge directly with a vLLM engine and a tokenizer.
        
        Args:
            llm: The initialized vLLM engine instance.
            tokenizer: The corresponding tokenizer.
        """
        self.llm = llm
        self.tokenizer = tokenizer
        self.judge_sampling_params = SamplingParams(temperature=0.0, max_tokens=16)

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
        return float(match.group(0)) if match else 0.0

    def _evaluate(self, batch_df: pd.DataFrame, prompt_builder) -> pd.Series:
        """Internal method to run batch evaluation using the vLLM engine."""
        user_prompts = [
            prompt_builder(
                question=row.questions,
                answer=row.result,
                context=row.source_documents
            )
            for row in batch_df.itertuples()
        ]
        
        full_prompts = [self._build_prompt(p) for p in user_prompts]
        
        # vLLM `generate` is optimized for batching prompts
        outputs = self.llm.generate(full_prompts, self.judge_sampling_params)
        
        # Extract text and then the score from each output
        responses_text = [output.outputs[0].text.strip() for output in outputs]
        scores = [self._extract_score(res) for res in responses_text]
        
        return pd.Series(scores, index=batch_df.index, dtype=float)

    def evaluate_faithfulness(self, batch_df: pd.DataFrame) -> pd.Series:
        """
        Scores how faithful the answer is to the provided context (i.e., groundedness).
        A high score means the answer contains no hallucinations or information unsupported by the context.
        """
        def prompt_builder(question, answer, context):
            return (
                "Evaluate the factual alignment of the 'Answer' strictly based on the provided 'Context'.\n\n"
                f"**Context:**\n---\n{context}\n---\n\n"
                f"**Question:** {question}\n\n"
                f"**Answer:** {answer}\n\n"
                "**Scoring Guidelines (Faithfulness):**\n"
                "- **1.0 (Perfectly Faithful):** Every single claim made in the 'Answer' is explicitly supported and verifiable by the 'Context'.\n"
                "- **0.0 (Not Faithful):** The 'Answer' contradicts the 'Context' or is a complete hallucination with no basis in the 'Context'.\n"
                "- **Intermediate Score:** The 'Answer' is mostly faithful but contains minor claims not found in the 'Context'. The score should decrease as the number and severity of unsupported claims increase. For example, an answer that is 90% supported by the context but includes one minor unsubstantiated detail might score **0.85**.\n\n"
            )
        return self._evaluate(batch_df, prompt_builder)

    def evaluate_relevance(self, batch_df: pd.DataFrame) -> pd.Series:
        """
        Scores how relevant the answer is to the user's question.
        A high score means the answer directly and completely addresses the user's intent.
        """
        def prompt_builder(question, answer, context):
            # Context is ignored for relevance but included for function signature consistency
            return (
                "Evaluate how well the 'Answer' directly addresses the user's 'Question'.\n\n"
                f"**Question:** {question}\n\n"
                f"**Answer:** {answer}\n\n"
                "**Scoring Guidelines (Relevance):**\n"
                "- **1.0 (Perfectly Relevant):** The 'Answer' directly, completely, and concisely addresses all parts of the 'Question'. It fully satisfies the user's intent.\n"
                "- **0.0 (Not Relevant):** The 'Answer' is completely off-topic or fails to address the 'Question' in any meaningful way.\n"
                "- **Intermediate Score:** The 'Answer' is on-topic but only partially addresses the 'Question', misses a key aspect, or includes redundant information. For example, an answer that addresses the main part of a two-part question but ignores the second part might score **0.6**.\n\n"
            )
        return self._evaluate(batch_df, prompt_builder)

    def evaluate_conciseness(self, batch_df: pd.DataFrame) -> pd.Series:
        """
        Scores how concise the answer is.
        A high score means the answer addresses the question without unnecessary verbosity or repetition.
        """
        def prompt_builder(question, answer, context):
            # Context is ignored for conciseness
            return (
                "Evaluate the conciseness of the 'Answer' in relation to the 'Question'.\n\n"
                f"**Question:** {question}\n\n"
                f"**Answer:** {answer}\n\n"
                "**Scoring Guidelines (Conciseness):**\n"
                "- **1.0 (Perfectly Concise):** The 'Answer' is direct, to the point, and contains no filler or redundant information.\n"
                "- **0.0 (Not Concise):** The 'Answer' is extremely verbose, repetitive, and contains significant information not relevant to the user's 'Question'.\n"
                "- **Intermediate Score:** The 'Answer' is mostly direct but contains some minor verbosity or slightly off-topic details. For example, an answer that fully addresses the question but includes a few unnecessary sentences might score **0.7**.\n\n"
            )
        return self._evaluate(batch_df, prompt_builder)


