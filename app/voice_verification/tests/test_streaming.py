from __future__ import annotations

import asyncio

from app.assistant.providers.base import LLMProvider
from app.assistant.streaming import SentenceBuffer


def test_sentence_buffer_releases_complete_sentences_and_flushes_tail() -> None:
    buffer = SentenceBuffer(maximum_characters=80)

    assert buffer.push("Xin chào. Đây là") == ["Xin chào."]
    assert buffer.push(" câu thứ hai") == []
    assert buffer.flush() == ["Đây là câu thứ hai"]


def test_sentence_buffer_has_a_bounded_fallback_when_the_model_omits_punctuation() -> None:
    buffer = SentenceBuffer(maximum_characters=12)

    assert buffer.push("một hai ba bốn năm") == ["một hai ba"]
    assert buffer.flush() == ["bốn năm"]


class _FallbackProvider(LLMProvider):
    async def respond(self, transcript, *, allowed_tools, auth_context) -> str:
        return "final answer"


def test_llm_stream_default_preserves_the_existing_non_streaming_provider_contract() -> None:
    async def collect() -> list[str]:
        provider = _FallbackProvider()
        return [chunk async for chunk in provider.respond_stream("hello", allowed_tools=set(), auth_context=None)]

    assert asyncio.run(collect()) == ["final answer"]
