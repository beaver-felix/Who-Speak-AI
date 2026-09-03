"""Typed calendar provider boundary; mock first, MCP-compatible later."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from app.assistant_gateway.store import CalendarEvent, GatewayStore


@dataclass(frozen=True)
class CalendarResult:
    provider: str
    demo: bool
    user_id: str
    events: tuple[CalendarEvent, ...]


class CalendarProvider(Protocol):
    async def list_events(self, *, user_id: str, start: datetime, end: datetime) -> CalendarResult: ...

    async def create_event(self, *, user_id: str, title: str, start: datetime, end: datetime) -> CalendarResult: ...


class MockCalendarProvider:
    """Deterministic local data. It never contacts Google or an MCP server."""

    def __init__(self, store: GatewayStore) -> None:
        self._store = store

    async def list_events(self, *, user_id: str, start: datetime, end: datetime) -> CalendarResult:
        events = self._store.list_mock_events(user_id, start=start, end=end)
        if not events:
            demo_start = datetime(2026, 8, 24, 14, 0, tzinfo=UTC)
            demo_end = demo_start + timedelta(hours=1)
            events = [CalendarEvent("demo-event-1", user_id, "Demo meeting", demo_start, demo_end)]
        return CalendarResult("mock", True, user_id, tuple(events))

    async def create_event(self, *, user_id: str, title: str, start: datetime, end: datetime) -> CalendarResult:
        if not title.strip() or end <= start:
            raise ValueError("A calendar event needs a title and a valid time range.")
        event = self._store.create_mock_event(user_id=user_id, title=title, start=start, end=end)
        return CalendarResult("mock", True, user_id, (event,))
