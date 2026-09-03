from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.assistant.agents.conversation import ConversationSupervisor
from app.assistant.auth_session import AuthSession
from app.assistant.contracts import AuthDecision, AuthState
from app.assistant.tools.calendar import CalendarResult
from app.assistant_gateway.store import CalendarEvent
from voiceauth.matching import VerificationResult


class FakeLLM:
    def __init__(self) -> None:
        self.allowed_tools: set[str] | None = None
        self.prompts: list[str] = []

    async def respond(self, transcript, *, allowed_tools, auth_context) -> str:
        self.allowed_tools = allowed_tools
        self.prompts.append(transcript)
        return "safe response"


class FakeCalendar:
    def __init__(self) -> None:
        self.calls = 0
        self.user_ids: list[str] = []
        self.ranges: list[tuple[datetime, datetime]] = []

    async def list_events(self, *, user_id, start, end) -> CalendarResult:
        self.calls += 1
        self.user_ids.append(user_id)
        self.ranges.append((start, end))
        event = CalendarEvent("event", user_id, "Demo meeting", start, end)
        return CalendarResult("mock", True, user_id, (event,))

    async def create_event(self, *, user_id, title, start, end) -> CalendarResult:
        self.calls += 1
        self.user_ids.append(user_id)
        event = CalendarEvent("created", user_id, title, start, end)
        return CalendarResult("mock", True, user_id, (event,))


def test_guest_calendar_request_requires_voice_auth_without_calling_llm_or_calendar() -> None:
    llm, calendar = FakeLLM(), FakeCalendar()
    supervisor = ConversationSupervisor(llm=llm, calendar=calendar, account_id="account-guest")

    response, tool_result = asyncio.run(
        supervisor.respond("Lịch hôm nay của tôi?", auth=AuthSession(ttl=timedelta(minutes=5)).current())
    )

    assert response == (
        "Bạn hãy xác thực voice trước để mình có thể xem lịch cá nhân. "
        "Sau khi xác thực xong, hãy hỏi lại mình."
    )
    assert tool_result is None
    assert calendar.calls == 0
    assert llm.allowed_tools is None
    assert llm.prompts == []


def test_expired_voice_session_requires_reauthentication_before_calendar_access() -> None:
    llm, calendar = FakeLLM(), FakeCalendar()
    supervisor = ConversationSupervisor(llm=llm, calendar=calendar, account_id="account-expired")

    response, tool_result = asyncio.run(
        supervisor.respond("Xem lịch cá nhân của tôi", auth=AuthDecision(AuthState.SESSION_EXPIRED))
    )

    assert response == (
        "Phiên xác thực voice đã hết hạn. Bạn hãy xác thực voice lại trước "
        "để mình có thể xem lịch cá nhân."
    )
    assert tool_result is None
    assert calendar.calls == 0
    assert llm.prompts == []


def test_authenticated_calendar_request_is_explicitly_marked_as_mock() -> None:
    llm, calendar = FakeLLM(), FakeCalendar()
    session = AuthSession(ttl=timedelta(minutes=5))
    identity = str(uuid4())
    session.begin_private_auth()
    decision = session.complete_verification(VerificationResult(True, identity, "An", 0.9, 1, identity))
    assert decision.state is AuthState.AUTHENTICATED
    supervisor = ConversationSupervisor(llm=llm, calendar=calendar, account_id="account-123")

    _, tool_result = asyncio.run(supervisor.respond("Lịch hôm nay của tôi?", auth=decision))

    assert calendar.calls == 1
    assert calendar.user_ids == ["account-123"]
    assert tool_result is not None
    assert tool_result["provider"] == "mock"
    assert tool_result["demo"] is True
    assert llm.allowed_tools == {"general_qa", "calendar.list_events", "calendar.create_event"}


def test_authenticated_today_request_resolves_the_window_before_calendar_call() -> None:
    llm, calendar = FakeLLM(), FakeCalendar()
    session = AuthSession(ttl=timedelta(minutes=5))
    identity = str(uuid4())
    session.begin_private_auth()
    decision = session.complete_verification(VerificationResult(True, identity, "An", 0.9, 1, identity))
    supervisor = ConversationSupervisor(
        llm=llm,
        calendar=calendar,
        account_id="account-today",
        timezone_name="Asia/Ho_Chi_Minh",
        clock=lambda: datetime(2026, 9, 3, 5, 30, tzinfo=UTC),
    )

    response, tool_result = asyncio.run(
        supervisor.respond("Bạn xem lịch hôm nay giúp tôi", auth=decision)
    )

    assert response == "safe response"
    assert tool_result is not None
    assert calendar.ranges == [
        (
            datetime(2026, 9, 2, 17, 0, tzinfo=UTC),
            datetime(2026, 9, 3, 17, 0, tzinfo=UTC),
        )
    ]
    assert "Trusted current time context" in llm.prompts[0]
    assert "'label': 'today'" in llm.prompts[0]


def test_authenticated_create_request_is_a_local_demo_event() -> None:
    llm, calendar = FakeLLM(), FakeCalendar()
    session = AuthSession(ttl=timedelta(minutes=5))
    identity = str(uuid4())
    session.begin_private_auth()
    decision = session.complete_verification(VerificationResult(True, identity, "An", 0.9, 1, identity))
    supervisor = ConversationSupervisor(llm=llm, calendar=calendar, account_id="account-456")

    _, tool_result = asyncio.run(supervisor.respond("Tạo lịch họp ngày mai", auth=decision))

    assert tool_result is not None
    assert tool_result["provider"] == "mock"
    assert tool_result["demo"] is True
    assert tool_result["created"]["title"].startswith("Demo request:")
    assert calendar.user_ids == ["account-456"]
