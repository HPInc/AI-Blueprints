
from __future__ import annotations

import logging, re, pandas as pd, torch, mlflow
from pathlib import Path
from typing import Union

from transformers import AutoTokenizer, AutoModelForCausalLM
from mlflow.types import Schema, ColSpec
from mlflow.models import ModelSignature
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

_FULL_WT_RGX = re.compile(r"^(pytorch_model|model)(-\d+)?\.(bin|safetensors)$")


def _dir_has_full_weights(path: Path) -> bool:
    return path.is_dir() and any(
        _FULL_WT_RGX.match(p.name) for p in path.iterdir() if p.is_file()
    )


def _as_path(obj) -> Path:
    """Ensure `obj` is Path (do not convert Hub IDs).."""
    return obj if isinstance(obj, Path) else Path(obj)


def _load_tokenizer(src: Union[str, Path]):
    p = _as_path(src)
    if p.exists():
        return AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    hub_id = str(src).replace("__", "/")
    return AutoTokenizer.from_pretrained(hub_id, trust_remote_code=True)


def _load_model(
    src: Union[str, Path],
    device: str,
    dtype: str = "auto",
    trust_remote: bool = True,
):
    p = _as_path(src)
    if p.exists():
        return AutoModelForCausalLM.from_pretrained(
            str(p),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote,
        ).to(device)

    hub_id = str(src).replace("__", "/")
    logging.info("🌐 Downloading from Hub: %s", hub_id)
    return AutoModelForCausalLM.from_pretrained(
        hub_id,
        torch_dtype=dtype,
        trust_remote_code=trust_remote,
    ).to(device)


class LLMComparisonModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        base_src = context.artifacts["model_no_finetuning"]
        ft_src   = context.artifacts["finetuned_model"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer_base = _load_tokenizer(base_src)
        self.model_base     = _load_model(base_src, self.device).eval()

        ft_path = _as_path(ft_src)
        if ft_path.exists() and ft_path.is_dir():
            if _dir_has_full_weights(ft_path):
                logging.info("🟢 Fine-tuned = complete checkpoint")
                self.tokenizer_ft = _load_tokenizer(ft_path)
                self.model_ft     = _load_model(ft_path, self.device).eval()
            else:
                logging.info("🟠 Fine-tuned = LoRA adapter only")
                self.tokenizer_ft = self.tokenizer_base
                self.model_ft = (
                    PeftModel.from_pretrained(
                        self.model_base, str(ft_path), is_trainable=False
                    )
                    .merge_and_unload()
                    .to(self.device)
                    .eval()
                )
        else:
            logging.info("🌐 Fine-tuned = : %s", ft_src)
            self.tokenizer_ft = _load_tokenizer(ft_src)
            self.model_ft     = _load_model(ft_src, self.device).eval()

        self.current_model = self.current_tok = None

    def _select(self, use_ft: bool):
        if use_ft:
            self.current_model, self.current_tok = self.model_ft, self.tokenizer_ft
        else:
            self.current_model, self.current_tok = self.model_base, self.tokenizer_base

    def predict(self, context, X: pd.DataFrame) -> pd.DataFrame:
        prompt   = X["prompt"].iloc[0]
        use_ft   = X["use_finetuning"].iloc[0]
        max_tok  = X.get("max_tokens", 128).iloc[0] if "max_tokens" in X else 128

        self._select(use_ft)

        inputs = self.current_tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ids = self.current_model.generate(
                **inputs,
                max_new_tokens=max_tok,
                do_sample=False,
                pad_token_id=self.current_tok.eos_token_id,
            )

        txt = self.current_tok.decode(ids[0], skip_special_tokens=True)
        return pd.DataFrame({"response": [txt]})


def register_llm_comparison_model(
    model_base_path: str,
    model_finetuned_path: str,
    experiment: str,
    run_name: str,
    registry_name: str,
):
    core = Path(__file__).resolve().parent.parent
    (core / "__init__.py").touch(exist_ok=True)

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        signature = ModelSignature(
            inputs=Schema(
                [
                    ColSpec("string",  "prompt"),
                    ColSpec("boolean", "use_finetuning"),
                    ColSpec("integer", "max_tokens"),
                ]
            ),
            outputs=Schema([ColSpec("string", "response")]),
        )

        mlflow.pyfunc.log_model(
            artifact_path="llm_serving_model",
            python_model=LLMComparisonModel(),
            artifacts={
                "model_no_finetuning": model_base_path,
                "finetuned_model":     model_finetuned_path,
            },
            signature=signature,
            code_paths=[str(core)],
            pip_requirements=[
                "torch",
                "transformers==4.51.3",
                "peft==0.15.2",
                "accelerate==1.6.0",
                "mlflow",
                "pandas",
            ],
        )

        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/llm_serving_model",
            name=registry_name,
        )
        logging.info("✅ Registered as `%s` (run %s)", registry_name, run.info.run_id)
