"""Small, policy-first conversational orchestrator.

The LLM is asked for natural-language output only. Tool eligibility and tool
execution remain deterministic Python code on the trusted side of the Agent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

from app.assistant.agents.supervisor import SupervisorPolicy
from app.assistant.contracts import AuthDecision
from app.assistant.pipecat_runtime.tools import PolicyToolExecutor
from app.assistant.tools.calendar import CalendarProvider


class ConversationSupervisor:
    def __init__(
        self,
        *,
        llm,
        calendar: CalendarProvider | None = None,
        account_id: str | None = None,
    ) -> None:
        self._llm = llm
        self._tools = PolicyToolExecutor(calendar, account_id=account_id)

    async def _prepare(self, transcript: str, *, auth: AuthDecision) -> tuple[str, set[str], dict | None]:
        allowed_tools = SupervisorPolicy.for_auth(auth)
        normalized = transcript.casefold()
        tool_result: dict | None = None

        asks_for_calendar = any(keyword in normalized for keyword in ("lịch", "calendar", "sự kiện"))
        asks_to_create = any(keyword in normalized for keyword in ("tạo", "đặt", "create", "add"))
        if "calendar.create_event" in allowed_tools and asks_for_calendar and asks_to_create:
            now = datetime.now(UTC).replace(second=0, microsecond=0)
            tool_result = await self._tools.create_event(
                auth=auth,
                # This intentionally does not pretend to understand a calendar
                # date. Phase 4 is a clearly-labelled demo provider; real time
                # extraction comes with the MCP/OAuth phase.
                title=f"Demo request: {transcript.strip()[:120]}",
                start=now + timedelta(hours=1),
                end=now + timedelta(hours=2),
            )
        elif "calendar.list_events" in allowed_tools and asks_for_calendar:
            now = datetime.now(UTC)
            tool_result = await self._tools.list_events(
                auth=auth, start=now, end=now + timedelta(days=7)
            )

        prompt = transcript.strip()
        if tool_result is not None:
            prompt += f"\n\nCalendar tool result (demo data): {tool_result}"
        return prompt, allowed_tools, tool_result

    async def respond_stream(self, transcript: str, *, auth: AuthDecision) -> AsyncIterator[tuple[str, dict | None]]:
        """Yield trusted LLM text deltas after deterministic tool policy execution."""
        prompt, allowed_tools, tool_result = await self._prepare(transcript, auth=auth)
        stream = getattr(self._llm, "respond_stream", None)
        if callable(stream):
            async for chunk in stream(prompt, allowed_tools=allowed_tools, auth_context=auth):
                if chunk:
                    yield chunk, tool_result
            return
        response = await self._llm.respond(prompt, allowed_tools=allowed_tools, auth_context=auth)
        if response:
            yield response, tool_result

    async def respond(self, transcript: str, *, auth: AuthDecision) -> tuple[str, dict | None]:
        chunks: list[str] = []
        tool_result: dict | None = None
        async for chunk, streamed_tool_result in self.respond_stream(transcript, auth=auth):
            chunks.append(chunk)
            tool_result = streamed_tool_result
        response = "".join(chunks).strip()
        if not response:
            raise RuntimeError("The language model returned no text response.")
        return response, tool_result
