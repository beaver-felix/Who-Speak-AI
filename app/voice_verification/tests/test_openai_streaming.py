from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.assistant.config import OpenAISettings
from app.assistant.contracts import AuthDecision, AuthState
from app.assistant.providers.openai_llm import OpenAIResponsesProvider


class _Events:
    def __init__(self) -> None:
        self._values = [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_text.delta", delta="Xin "),
            SimpleNamespace(type="response.output_text.delta", delta="chào"),
            SimpleNamespace(type="response.completed"),
        ]

    def __aiter__(self):
        self._iterator = iter(self._values)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration as error:
            raise StopAsyncIteration from error


class _Responses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Events()


class _Client:
    def __init__(self) -> None:
        self.responses = _Responses()


def test_openai_provider_yields_only_visible_text_deltas_and_keeps_store_disabled() -> None:
    client = _Client()
    provider = OpenAIResponsesProvider(OpenAISettings(api_key="test", model="test-model"), client=client)

    async def collect() -> list[str]:
        return [
            chunk
            async for chunk in provider.respond_stream(
                "Xin chào", allowed_tools={"general_qa"}, auth_context=AuthDecision(AuthState.GUEST)
            )
        ]

    assert asyncio.run(collect()) == ["Xin ", "chào"]
    assert len(client.responses.calls) == 1
    request = client.responses.calls[0]
    assert request["model"] == "test-model"
    assert request["input"] == "Xin chào"
    assert request["store"] is False
    assert request["stream"] is True
    assert "Trusted VoiceAuth state for this request: guest" in request["instructions"]
    assert "Private Calendar access: not granted" in request["instructions"]
