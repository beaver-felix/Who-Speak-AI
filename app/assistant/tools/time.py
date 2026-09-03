"""Trusted current-time context for date-aware voice tools.

The LLM never supplies the timestamps used by Calendar.  This module resolves
relative phrases such as ``hôm nay`` and ``ngày mai`` in Python, using one
validated timezone, before a calendar provider is called.
"""

from __future__ import annotations

import os
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DEFAULT_TIMEZONE = "Asia/Ho_Chi_Minh"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _fold_text(value: str) -> str:
    """Normalize Vietnamese accents so both accented and unaccented input work."""
    normalized = unicodedata.normalize("NFD", value.casefold())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


@dataclass(frozen=True)
class CurrentDateTime:
    """A safe, serializable snapshot of the trusted application clock."""

    timezone: str
    local: datetime
    utc: datetime

    def as_prompt_context(self) -> dict[str, str]:
        return {
            "timezone": self.timezone,
            "iso": self.local.isoformat(),
            "date": self.local.date().isoformat(),
            "time": self.local.strftime("%H:%M"),
        }


@dataclass(frozen=True)
class CalendarTimeWindow:
    """A provider-ready UTC window plus the local label used in the prompt."""

    label: str
    timezone: str
    start: datetime
    end: datetime

    def as_prompt_context(self) -> dict[str, str]:
        return {
            "label": self.label,
            "timezone": self.timezone,
            "start_utc": self.start.isoformat(),
            "end_utc": self.end.isoformat(),
        }


class CurrentTimeTool:
    """Provide current time and deterministic relative-date resolution.

    This is an Agent-side Python tool, not an LLM function exposed to the
    browser.  The optional clock makes boundary behavior deterministic in
    tests and keeps all production calls timezone-aware.
    """

    def __init__(
        self,
        *,
        timezone_name: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        selected_timezone = (
            timezone_name or os.getenv("VOICE_DEFAULT_TIMEZONE", DEFAULT_TIMEZONE)
        ).strip()
        if not selected_timezone:
            selected_timezone = DEFAULT_TIMEZONE
        try:
            self._zone = ZoneInfo(selected_timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"Unknown voice timezone: {selected_timezone}") from error
        self._timezone = selected_timezone
        self._clock = clock or _utc_now

    @property
    def timezone(self) -> str:
        return self._timezone

    def get_current_datetime(self) -> CurrentDateTime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("The current-time clock must return an aware datetime.")
        utc = value.astimezone(UTC)
        return CurrentDateTime(
            timezone=self._timezone,
            local=utc.astimezone(self._zone),
            utc=utc,
        )

    def calendar_window(
        self,
        transcript: str,
        *,
        current: CurrentDateTime | None = None,
    ) -> CalendarTimeWindow:
        snapshot = current or self.get_current_datetime()
        normalized = _fold_text(transcript)
        local_now = snapshot.local

        if "hom nay" in normalized or "today" in normalized:
            start_local = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_local = start_local + timedelta(days=1)
            label = "today"
        elif "ngay mai" in normalized or "tomorrow" in normalized:
            start_local = local_now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            end_local = start_local + timedelta(days=1)
            label = "tomorrow"
        elif "hom qua" in normalized or "yesterday" in normalized:
            end_local = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
            start_local = end_local - timedelta(days=1)
            label = "yesterday"
        else:
            # Preserve the previous product behavior for an unspecified
            # calendar request: upcoming events from now through seven days.
            start_local = local_now
            end_local = local_now + timedelta(days=7)
            label = "next_7_days"

        return CalendarTimeWindow(
            label=label,
            timezone=self._timezone,
            start=start_local.astimezone(UTC),
            end=end_local.astimezone(UTC),
        )


def get_current_datetime(
    timezone_name: str | None = None,
    *,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, str]:
    """Return current local date/time for a trusted Agent-side tool call."""
    return CurrentTimeTool(
        timezone_name=timezone_name, clock=clock
    ).get_current_datetime().as_prompt_context()
