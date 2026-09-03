"""Provider interfaces that never accept private HE data or raw audio."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.assistant.contracts import AuthDecision


class LLMProvider(ABC):
    @abstractmethod
    async def respond(
        self,
        transcript: str,
        *,
        allowed_tools: set[str],
        auth_context: AuthDecision,
    ) -> str:
        """Return text for a transcript after the trusted policy selected capabilities."""

    async def respond_stream(
        self,
        transcript: str,
        *,
        allowed_tools: set[str],
        auth_context: AuthDecision,
    ) -> AsyncIterator[str]:
        """Yield text deltas while retaining compatibility with batch providers."""
        yield await self.respond(transcript, allowed_tools=allowed_tools, auth_context=auth_context)
