
from __future__ import annotations

import logging
from typing import Callable, List, Optional

from langchain.schema.runnable import Runnable

try:  
    import promptquality as pq  
except ModuleNotFoundError: 
    pq = None  

__all__ = ["ScriptGenerator"]


class ScriptGenerator:
    """
    Parameters
    ----------
    chain : Runnable
        A LangChain runnable that will be invoked for each section.
    scorers : list[Callable] | None, optional
        Explicit list of Prompt-Quality scorers. If *None*, a sensible default
        is used when Galileo is active.
    use_galileo : bool, default ``True``
        Whether to log Prompt-Quality metrics to Galileo.
    logging_enabled : bool, default ``False``
        Verbose, per-section logging to ``stdout`` and the module logger.
    """

    def __init__(
        self,
        chain: Runnable,
        scorers: Optional[List[Callable]] = None,
        *,
        use_galileo: bool = True,
        logging_enabled: bool = False,
    ):
        self.chain: Runnable = chain
        self.sections: List[dict[str, str]] = []
        self.results: dict[str, str] = {}
        self.use_galileo: bool = bool(use_galileo)

        if self.use_galileo and pq is None:
            logging.warning("promptquality not installed – Galileo disabled.")
            self.use_galileo = False

        self.scorers: List[Callable] = (
            scorers
            if scorers is not None
            else (
                [
                    pq.Scorers.context_adherence_plus,
                    pq.Scorers.correctness,
                    pq.Scorers.prompt_perplexity,
                ]
                if self.use_galileo
                else []
            )
        )

        self.logger = logging.getLogger(__name__)
        self.logging_enabled = logging_enabled
        if logging_enabled:
            logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    def add_section(self, name: str, prompt: str) -> None:
        """Append a named section that will be generated later."""
        self.sections.append({"name": name, "prompt": prompt})
        if self.logging_enabled:
            self.logger.info("Section '%s' added.", name)

    def run(self) -> None:
        """Generate every registered section, storing the approved result."""
        for section in self.sections:
            if self.logging_enabled:
                self.logger.info("Running section '%s'.", section["name"])
            self.results[section["name"]] = self._run_and_approve(section)

    def get_final_script(self) -> str:
        """Return the concatenation of all approved sections."""
        return "\n\n".join(self.results.values())

    def _run_and_approve(self, section: dict[str, str]) -> str:
        """
        Generate content for *one* section until the user approves it
        (or approves on the first try). Blocks for interactive feedback.
        """
        while True:
            # 1) Optional Galileo callback
            prompt_handler = None
            callbacks: list = []
            if self.use_galileo:
                prompt_handler = pq.GalileoPromptCallback(scorers=self.scorers)  # type: ignore
                callbacks = [prompt_handler]

            # 2) Execute the runnable
            if self.logging_enabled:
                self.logger.info("Generating section '%s'…", section["name"])
            result_batch = self.chain.batch(
                [{"prompt": section["prompt"]}],
                config=dict(callbacks=callbacks),
            )
            raw_text: str = result_batch[0] if result_batch else ""

            if self.logging_enabled:
                preview = raw_text[:300].replace("\n", " ") + ("…" if len(raw_text) > 300 else "")
                self.logger.info("Model output (%s): %s", section["name"], preview)

            # 3) Interactive approval loop
            print(f"\n>>> [{section['name']}] Result:\n{raw_text}\n")
            approval = input("Approve the result? (y/n): ").strip().lower()
            if approval == "y":
                if prompt_handler is not None:
                    prompt_handler.finish()  # push metrics to Galileo
                return raw_text

            print("Result not approved – regenerating…\n")
