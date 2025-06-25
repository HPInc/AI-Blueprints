# src/local_judge.py
import re, json
from typing import Dict, Any, List
from opik.evaluation.models import OpikBaseModel
from langchain_community.llms import LlamaCpp

JSON_GRAMMAR = r'''
root    ::= object
object  ::= "{" members? "}"
members ::= pair ("," pair)*
pair    ::= string ":" value
string  ::= "\"" [^"]* "\""
value   ::= string | number | bool
number  ::= [0-9]+ ("." [0-9]+)?
bool    ::= "true" | "false"
'''


class LangChainJudge(OpikBaseModel):
    """
    Loads a *separate* llama-cpp instance using paths & params from
    your existing `config` dict, so the main `llm` stays untouched.
    """

    _mini_json = re.compile(r"\{.*?\}", re.S)


    def __init__(self, model_path: str):
        # • deterministic, JSON-safe decoding
        self.llm = LlamaCpp(
            model_path      = model_path,
            n_gpu_layers    = -1,
            n_ctx           = 4096,
            temperature     = 0.0,
            top_p           = 1.0,
            max_tokens      = 64,
            stop            = ["}"],
            grammar         = JSON_GRAMMAR,
            f16_kv          = True,
            streaming       = False,
            verbose         = False,
            model_kwargs    = {"chat_format": "llama-3"},
        )

    # ---------- internal helpers --------------------------
    def _clean(self, txt: str) -> str:
        print(txt)
        for m in self._mini_json.finditer(txt):
            candidate = m.group(0)
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                continue
        return '{"score": 1.0, "reason": "No valid JSON in judge output"}'

        
    def _run(self, prompt: str) -> str:
        return self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else self.llm(prompt)

    # ---------- Opik-required API -------------------------
    def generate_string(self, input: str, **_) -> str:
        return self._clean(self._run(input))

    def generate_provider_response(self, messages: List[Dict[str, Any]], **_) -> Any:
        prompt  = "\n".join(m["content"] for m in messages if m["role"] == "user")
        content = self.generate_string(prompt)

        class _M: pass; _m=_M(); _m.content = content
        class _C: pass; _c=_C(); _c.message = _m
        class _R: pass; _r=_R(); _r.choices = [_c]
        return _r
