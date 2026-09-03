"""Minimal OpenAI Responses API adapter for the local-first agent."""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.assistant.config import OpenAISettings
from app.assistant.contracts import AuthDecision
from app.assistant.providers.base import LLMProvider


class OpenAIResponsesProvider(LLMProvider):
    """Send transcripts only; no audio, embeddings, HE material, or API key logs."""

    def __init__(self, settings: OpenAISettings, *, client=None) -> None:
        self._settings = settings
        self._client = client

    def _client_or_create(self):
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except ImportError as error:
            raise RuntimeError("Install the [providers] extra to enable the OpenAI LLM provider.") from error
        self._client = AsyncOpenAI(api_key=self._settings.api_key)
        return self._client

    def _request_arguments(
        self,
        transcript: str,
        *,
        allowed_tools: set[str],
    ) -> dict[str, object]:
        text = transcript.strip()
        if not text:
            raise ValueError("A non-empty local ASR transcript is required.")
        instructions = (
            "You are a helpful voice assistant. Keep answers concise and natural when spoken. "
            "Follow the capability policy. "
            "Use trusted current-time context and Calendar tool results supplied by the application; "
            "never invent dates, times, events, or successful Calendar access. "
            f"Capabilities available for this request: {', '.join(sorted(allowed_tools)) or 'none'}."
        )
        return {
            "model": self._settings.model,
            "instructions": instructions,
            "input": text,
            "store": False,
        }

    async def respond(
        self,
        transcript: str,
        *,
        allowed_tools: set[str],
        auth_context: AuthDecision,
    ) -> str:
        # The capability list is selected by SupervisorPolicy before this call.
        # Do not include identity, score, embedding, HE context, or raw audio.
        response = await self._client_or_create().responses.create(
            **self._request_arguments(transcript, allowed_tools=allowed_tools),
        )
        output = getattr(response, "output_text", "")
        if not isinstance(output, str) or not output.strip():
            raise RuntimeError("OpenAI returned no text response.")
        return output.strip()

    async def respond_stream(
        self,
        transcript: str,
        *,
        allowed_tools: set[str],
        auth_context: AuthDecision,
    ) -> AsyncIterator[str]:
        """Yield only visible text deltas from the Responses API stream."""
        stream = await self._client_or_create().responses.create(
            **self._request_arguments(transcript, allowed_tools=allowed_tools),
            stream=True,
        )
        yielded = False
        async for event in stream:
            if getattr(event, "type", None) != "response.output_text.delta":
                continue
            delta = getattr(event, "delta", "")
            if isinstance(delta, str) and delta:
                yielded = True
                yield delta
        if not yielded:
            raise RuntimeError("OpenAI returned no text response.")
