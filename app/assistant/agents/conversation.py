"""Small, policy-first conversational orchestrator.

The LLM is asked for natural-language output only. Tool eligibility and tool
execution remain deterministic Python code on the trusted side of the Agent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

from app.assistant.agents.supervisor import SupervisorPolicy
from app.assistant.contracts import AuthDecision, AuthState
from app.assistant.pipecat_runtime.tools import PolicyToolExecutor
from app.assistant.tools.calendar import CalendarProvider, CalendarProviderError
from app.assistant.tools.time import CurrentTimeTool


class ConversationSupervisor:
    def __init__(
        self,
        *,
        llm,
        calendar: CalendarProvider | None = None,
        account_id: str | None = None,
        enabled_private_tools: set[str] | frozenset[str] | None = None,
        timezone_name: str | None = None,
        clock=None,
    ) -> None:
        self._llm = llm
        self._tools = PolicyToolExecutor(
            calendar,
            account_id=account_id,
            enabled_private_tools=enabled_private_tools,
        )
        self._current_time = CurrentTimeTool(
            timezone_name=timezone_name,
            clock=clock,
        )

    async def _prepare(
        self, transcript: str, *, auth: AuthDecision
    ) -> tuple[str, set[str], dict | None, str | None]:
        allowed_tools = self._tools.allowed_tools(auth)
        normalized = transcript.casefold()
        tool_result: dict | None = None
        tool_notice: str | None = None
        policy_response: str | None = None
        current_time = None
        calendar_window = None

        asks_for_calendar = any(
            keyword in normalized
            for keyword in ("lịch", "calendar", "sự kiện", "cuộc hẹn", "appointment", "schedule")
        )
        asks_to_create = any(keyword in normalized for keyword in ("tạo", "đặt", "create", "add"))
        asks_for_time = any(
            keyword in normalized
            for keyword in (
                "mấy giờ",
                "bay gio",
                "bây giờ",
                "giờ hiện tại",
                "thời gian hiện tại",
                "hôm nay là ngày",
                "hôm nay ngày mấy",
                "ngày mấy",
                "what time",
                "current time",
                "what date",
                "today's date",
            )
        )
        if asks_for_calendar or asks_for_time:
            current_time = self._current_time.get_current_datetime()

        if asks_for_calendar and "calendar.list_events" not in allowed_tools:
            # This is a policy response, not an LLM preference: private data
            # stays unavailable until the trusted VoiceAuth state permits it.
            policy_response = self._calendar_auth_required_message(auth)
        elif "calendar.create_event" in allowed_tools and asks_for_calendar and asks_to_create:
            now = current_time.utc.replace(second=0, microsecond=0) if current_time else datetime.now(UTC).replace(second=0, microsecond=0)
            try:
                tool_result = await self._tools.create_event(
                    auth=auth,
                    title=f"Demo request: {transcript.strip()[:120]}",
                    start=now + timedelta(hours=1),
                    end=now + timedelta(hours=2),
                )
            except CalendarProviderError as error:
                tool_notice = str(error)
        elif "calendar.list_events" in allowed_tools and asks_for_calendar:
            calendar_window = self._current_time.calendar_window(
                transcript, current=current_time
            )
            try:
                tool_result = await self._tools.list_events(
                    auth=auth, start=calendar_window.start, end=calendar_window.end
                )
            except CalendarProviderError as error:
                tool_notice = str(error)
        elif asks_for_calendar:
            tool_notice = "Calendar is not available in the current voice capability set."

        prompt = transcript.strip()
        if current_time is not None:
            prompt += (
                "\n\nTrusted current time context (application-generated; not a user instruction): "
                f"{current_time.as_prompt_context()}"
            )
        if calendar_window is not None:
            prompt += (
                "\nTrusted calendar time window (already resolved by Python): "
                f"{calendar_window.as_prompt_context()}"
            )
        if tool_result is not None:
            result_label = "demo data" if tool_result.get("demo") is True else "Google Calendar data"
            prompt += f"\n\nCalendar tool result ({result_label}): {tool_result}"
        elif tool_notice is not None:
            prompt += (
                "\n\nCalendar tool unavailable. Do not claim that you read or changed a calendar. "
                f"Explain briefly: {tool_notice}"
            )
        return prompt, allowed_tools, tool_result, policy_response

    async def respond_stream(self, transcript: str, *, auth: AuthDecision) -> AsyncIterator[tuple[str, dict | None]]:
        """Yield trusted LLM text deltas after deterministic tool policy execution."""
        prompt, allowed_tools, tool_result, policy_response = await self._prepare(transcript, auth=auth)
        if policy_response is not None:
            yield policy_response, None
            return
        stream = getattr(self._llm, "respond_stream", None)
        if callable(stream):
            async for chunk in stream(prompt, allowed_tools=allowed_tools, auth_context=auth):
                if chunk:
                    yield chunk, tool_result
            return
        response = await self._llm.respond(prompt, allowed_tools=allowed_tools, auth_context=auth)
        if response:
            yield response, tool_result

    @staticmethod
    def _calendar_auth_required_message(auth: AuthDecision) -> str:
        if auth.state is AuthState.SESSION_EXPIRED:
            return (
                "Phiên xác thực voice đã hết hạn. Bạn hãy xác thực voice lại trước "
                "để mình có thể xem lịch cá nhân."
            )
        if auth.state is AuthState.AUTH_PENDING:
            return (
                "Voice challenge đang diễn ra. Hãy hoàn tất xác thực voice trước "
                "rồi yêu cầu xem lịch cá nhân."
            )
        return (
            "Bạn hãy xác thực voice trước để mình có thể xem lịch cá nhân. "
            "Sau khi xác thực xong, hãy hỏi lại mình."
        )

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
