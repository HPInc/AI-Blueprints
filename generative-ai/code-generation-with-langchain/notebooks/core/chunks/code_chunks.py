
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union


# ─────────────────────────  Tokenizer  ────────────────────────── #
def _default_tokenizer(text: str) -> List[str]:
    """
    Very simple fallback tokenizer: splits on words and punctuation.
    Replace with `tiktoken` or any tokenizer you prefer for better accuracy.
    """
    return re.findall(r"\w+|[^\s\w]", text, re.UNICODE)


# ─────────────────────────  Main class  ───────────────────────── #
class CodeChunker:
    """
    Smart, AST-aware chunker (works best for Python; falls back to lines).
    """

    def __init__(
        self,
        max_tokens: int = 300,
        overlap_tokens: int = 50,
        tokenizer: Optional[Any] = None,          # Callable[[str], List[str]]
        allowed_ext: Tuple[str, ...] = (".py",),  # Used only in `chunk_repository`
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tokenizer or _default_tokenizer
        self.allowed_ext = allowed_ext

    # ───────────────────────  Public API  ─────────────────────── #

    def chunk_code(self, code: str) -> List[str]:
        """Chunk a single source-code string."""
        try:
            return self._chunk_code_ast(code)
        except (SyntaxError, ValueError):
            return self._chunk_code_lines(code)

    def chunk_file(self, filepath: Union[str, Path]) -> List[str]:
        """Read a file and chunk its contents."""
        with open(filepath, "r", encoding="utf-8") as f:
            return self.chunk_code(f.read())

    def chunk_repository(
        self,
        repo_path: Union[str, Path],
        glob: str = "**/*.py",
    ) -> Dict[str, List[str]]:
        """
        Walk a directory recursively and chunk every file whose suffix
        matches `allowed_ext`.
        """
        repo_path = Path(repo_path).expanduser().resolve()
        out: Dict[str, List[str]] = {}
        for path in repo_path.glob(glob):
            if path.suffix.lower() in self.allowed_ext:
                rel = str(path.relative_to(repo_path))
                out[rel] = self.chunk_file(path)
        return out

    # ──────────────────────  Internal logic  ───────────────────── #

    def _chunk_code_ast(self, code: str) -> List[str]:
        """
        AST-aware chunking: group entire top-level nodes
        (imports, functions, classes) until the token budget is hit.
        """
        tree = ast.parse(code)
        spans: List[Tuple[int, int]] = [
            (getattr(n, "lineno"), getattr(n, "end_lineno"))
            for n in tree.body
            if getattr(n, "lineno", None) and getattr(n, "end_lineno", None)
        ]
        if not spans:   # empty file or unparsable
            return self._chunk_code_lines(code)

        lines = code.splitlines()
        token = self.tokenizer

        chunks: List[str] = []
        buf: List[str] = []
        buf_tok = 0

        def flush() -> None:
            nonlocal buf, buf_tok
            if buf:
                chunks.append("\n".join(buf).rstrip())
            buf, buf_tok = [], 0

        for start, end in spans:
            segment = lines[start - 1 : end]
            seg_text = "\n".join(segment)
            seg_tok = len(token(seg_text))

            # If a single node is too big, fall back to line-based split
            if seg_tok > self.max_tokens:
                flush()
                chunks.extend(self._chunk_code_lines(seg_text))
                continue

            if buf_tok + seg_tok > self.max_tokens:
                flush()

            buf.extend(segment)
            buf_tok += seg_tok

        flush()

        return self._apply_overlap(chunks) if self._needs_overlap(chunks) else chunks

    def _chunk_code_lines(self, code: str) -> List[str]:
        """
        Simple line-based chunker: split when token budget is exceeded.
        """
        token = self.tokenizer
        chunks, buf = [], []
        buf_tok = 0

        for line in code.splitlines():
            ln_tok = len(token(line))
            if buf_tok + ln_tok > self.max_tokens:
                chunks.append("\n".join(buf).rstrip())
                buf, buf_tok = [], 0
            buf.append(line)
            buf_tok += ln_tok

        if buf:
            chunks.append("\n".join(buf).rstrip())

        return self._apply_overlap(chunks) if self._needs_overlap(chunks) else chunks

    # ─────────────────────────  Helpers  ───────────────────────── #

    def _needs_overlap(self, chunks: List[str]) -> bool:
        return self.overlap_tokens > 0 and len(chunks) > 1

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Add token-level overlap between consecutive chunks to preserve context.
        """
        token = self.tokenizer
        ov = self.overlap_tokens
        merged = [chunks[0]]

        for nxt in chunks[1:]:
            head = token(nxt)[:ov]
            tail = token(merged[-1])[-ov:]
            if head == tail:
                merged.append(nxt)
            else:
                merged.append(" ".join(head) + "\n" + nxt)

        return merged



