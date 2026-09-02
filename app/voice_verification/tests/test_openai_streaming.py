from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.assistant.config import OpenAISettings
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
                "Xin chào", allowed_tools={"general_qa"}, auth_context=None
            )
        ]

    assert asyncio.run(collect()) == ["Xin ", "chào"]
    assert client.responses.calls == [
        {
            "model": "test-model",
            "instructions": "You are a helpful voice assistant. Keep answers concise and natural when spoken. Follow the capability policy. Capabilities available for this request: general_qa.",
            "input": "Xin chào",
            "store": False,
            "stream": True,
        }
    ]
