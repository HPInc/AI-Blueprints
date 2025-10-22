from __future__ import annotations

import logging
from typing import Callable, List, Optional

from langchain.schema.runnable import Runnable

__all__ = ["ScriptGenerator"]


class ScriptGenerator:
    """
    Parameters
    ----------
    chain : Runnable
        A LangChain runnable that will be invoked for each section.
    scorers : list[Callable] | None, optional
        Explicit list of scorer functions (kept for compatibility, ignored).
    use_local_logging : bool, default ``True``
        Whether to log metrics locally (disabled - kept for compatibility).
    logging_enabled : bool, default ``False``
        Verbose, per-section logging to ``stdout`` and the module logger.
    """

    def __init__(
        self,
        chain: Runnable,
        scorers: Optional[List[Callable]] = None,
        *,
        use_local_logging: bool = True,
        logging_enabled: bool = False,
    ):
        self.chain: Runnable = chain
        self.sections: List[dict[str, str]] = []
        self.results: dict[str, str] = {}
        self.use_local_logging: bool = False  # Always disabled

        # Remove external logging dependency check
        self.scorers: List[Callable] = []

        self.logger = logging.getLogger(__name__)
        self.logging_enabled = logging_enabled
        if logging_enabled:
            logging.basicConfig(
                level=logging.INFO, format="[%(levelname)s] %(message)s"
            )

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
            # Execute the runnable (external callback removed)
            if self.logging_enabled:
                self.logger.info("Generating section '%s'…", section["name"])
            result_batch = self.chain.batch(
                [{"prompt": section["prompt"]}],
                config=dict(callbacks=[]),  # No callbacks
            )
            raw_text: str = result_batch[0] if result_batch else ""

            if self.logging_enabled:
                preview = raw_text[:300].replace("\n", " ") + (
                    "…" if len(raw_text) > 300 else ""
                )
                self.logger.info("Model output (%s): %s", section["name"], preview)

            # Interactive approval loop
            print(f"\n>>> [{section['name']}] Result:\n{raw_text}\n")
            return raw_text
