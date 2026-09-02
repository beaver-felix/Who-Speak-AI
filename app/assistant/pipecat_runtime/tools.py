"""Typed, authorization-first tool execution for the Pipecat bridge."""

from __future__ import annotations

from datetime import datetime

from app.assistant.agents.supervisor import SupervisorPolicy
from app.assistant.contracts import AuthDecision
from app.assistant.tools.calendar import CalendarProvider


class ToolAuthorizationError(PermissionError):
    """Raised when a request is outside the current trusted capability set."""


class PolicyToolExecutor:
    """Execute typed calendar operations for one trusted account principal.

    ``AuthDecision.identity_id`` is a voice-template identifier, not a
    database account ID.  Keeping the two values separate prevents a speaker
    verification identifier from becoming a storage namespace.
    """

    def __init__(self, calendar: CalendarProvider | None, *, account_id: str | None) -> None:
        self._calendar = calendar
        self._account_id = account_id.strip() if account_id else None

    async def list_events(self, *, auth: AuthDecision, start: datetime, end: datetime) -> dict:
        account_id = self._authorize("calendar.list_events", auth)
        return await self._execute_calendar(
            "calendar.list_events", auth=auth, operation=lambda: self._calendar.list_events(
                user_id=account_id, start=start, end=end
            )
        )

    async def create_event(
        self, *, auth: AuthDecision, title: str, start: datetime, end: datetime
    ) -> dict:
        account_id = self._authorize("calendar.create_event", auth)
        return await self._execute_calendar(
            "calendar.create_event", auth=auth, operation=lambda: self._calendar.create_event(
                user_id=account_id, title=title, start=start, end=end
            )
        )

    def _authorize(self, name: str, auth: AuthDecision) -> str:
        if name not in SupervisorPolicy.for_auth(auth) or not auth.may_use_private_tools:
            raise ToolAuthorizationError(f"Tool {name} is not available in the current auth state.")
        if self._calendar is None or not self._account_id:
            raise RuntimeError("A trusted account context is required for calendar access.")
        return self._account_id

    async def _execute_calendar(self, name: str, *, auth: AuthDecision, operation) -> dict:
        result = await operation()
        if result.provider != "mock" or result.demo is not True or result.user_id != self._account_id:
            raise RuntimeError("Calendar provider returned an unsafe result.")
        payload: dict = {"provider": result.provider, "demo": result.demo}
        if name == "calendar.create_event":
            event = result.events[0]
            payload["created"] = {
                "title": event.title,
                "start": event.start.isoformat(),
                "end": event.end.isoformat(),
            }
        else:
            payload["events"] = [
                {"title": event.title, "start": event.start.isoformat(), "end": event.end.isoformat()}
                for event in result.events
            ]
        return payload
