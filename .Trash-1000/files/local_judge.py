import json
import re
import logging
from typing import Any, Dict, List
from langchain_community.llms import LlamaCpp
from opik.evaluation.models import OpikBaseModel

logger = logging.getLogger("llama_judge")
logger.setLevel(logging.DEBUG)

_JSON_SCORE_RE = re.compile(
    r'"(?:score|answer_relevance_score|context_precision_score|context_recall_score)"\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)'
)

def extract_score(text: str) -> float:
    m = _JSON_SCORE_RE.search(text)
    if m:
        return float(m.group(1))
    else: 
        return -1.0
    return 0.0


class LangChainJudge(OpikBaseModel):
    def __init__(self, model_path: str, metric: str):
        super().__init__("llama-judge")
        self.metric = metric.lower()  # e.g., "hallucination", "answer_relevance", etc.

        self.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            temperature=0.0,
            top_p=1.0,
            max_tokens=512,
            f16_kv=True,
            streaming=False,
            verbose=False,
            model_kwargs={"chat_format": "llama-3"},
        )

    def generate_string(self, input: str, **_) -> str:
        score = self._evaluate(input)

        if self.metric == "answer_relevance":
            payload = {"answer_relevance_score": score}
        elif self.metric == "context_precision":
            payload = {"context_precision_score": score}
        elif self.metric == "context_recall":
            payload = {"context_recall_score": score}
        else:                       # "hallucination" or fallback
            payload = {"score": score}

        # always include a generic reason
        if score == -1:
            payload["reason"] = [
            "Model did not output in a correct format."
        ]
        else:
            payload["reason"] = [
                "Small local model – explanation unavailable."
            ]
        return json.dumps(payload)
        
    def _raw_generate(self, prompt: str) -> str:
        output = self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else self.llm(prompt)
        logger.debug(f"Raw model output:\n{output}")
        return output.strip()

    def _evaluate(self, prompt: str) -> float:
        raw_output = self._raw_generate(prompt)
        logger.debug(f"Raw Score:\n{extract_score(raw_output)}")
        return extract_score(raw_output)

    def generate_provider_response(
            self,
            messages: List[Dict[str, Any]],
            **_: Any,
        ) -> Dict[str, Any]:
            """
            Required only to satisfy OpikBaseModel’s abstractmethod.
            Opik’s judge metrics never call this, so a stub is fine.
            """
            raise NotImplementedError(
                "generate_provider_response is not used in this context. "
                "Call generate_string() instead."
            )
