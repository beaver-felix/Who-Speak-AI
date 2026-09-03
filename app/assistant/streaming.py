"""Small provider-neutral helpers for incremental voice output."""

from __future__ import annotations

import re


class SentenceBuffer:
    """Release complete, bounded phrases without exposing token-level UI churn."""

    _boundary = re.compile(r"[.!?。！？](?:\s|$)")

    def __init__(self, *, maximum_characters: int = 180) -> None:
        if maximum_characters < 8:
            raise ValueError("maximum_characters must be at least 8.")
        self._maximum_characters = maximum_characters
        self._pending = ""

    def push(self, text: str) -> list[str]:
        self._pending += text
        released: list[str] = []
        while self._pending:
            boundary = self._boundary.search(self._pending)
            if boundary is not None:
                candidate = self._pending[: boundary.end()].strip()
                self._pending = self._pending[boundary.end() :].lstrip()
                if candidate:
                    released.append(candidate)
                continue
            if len(self._pending) < self._maximum_characters:
                break
            split_at = self._pending.rfind(" ", 0, self._maximum_characters + 1)
            if split_at <= 0:
                split_at = self._maximum_characters
            candidate = self._pending[:split_at].strip()
            self._pending = self._pending[split_at:].lstrip()
            if candidate:
                released.append(candidate)
        return released

    def flush(self) -> list[str]:
        tail = self._pending.strip()
        self._pending = ""
        return [tail] if tail else []
