from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.assistant.tools.time import CurrentTimeTool, get_current_datetime


FIXED_NOW = datetime(2026, 9, 3, 5, 30, tzinfo=UTC)


def test_current_datetime_returns_local_and_machine_safe_context() -> None:
    context = get_current_datetime(
        "Asia/Ho_Chi_Minh", clock=lambda: FIXED_NOW
    )

    assert context == {
        "timezone": "Asia/Ho_Chi_Minh",
        "iso": "2026-09-03T12:30:00+07:00",
        "date": "2026-09-03",
        "time": "12:30",
    }


def test_today_window_uses_local_midnight_and_converts_to_utc() -> None:
    tool = CurrentTimeTool(
        timezone_name="Asia/Ho_Chi_Minh", clock=lambda: FIXED_NOW
    )

    window = tool.calendar_window("Xem lịch hôm nay")

    assert window.label == "today"
    assert window.start == datetime(2026, 9, 2, 17, 0, tzinfo=UTC)
    assert window.end == datetime(2026, 9, 3, 17, 0, tzinfo=UTC)


def test_unaccented_tomorrow_phrase_is_supported() -> None:
    tool = CurrentTimeTool(
        timezone_name="Asia/Ho_Chi_Minh", clock=lambda: FIXED_NOW
    )

    window = tool.calendar_window("xem lich ngay mai")

    assert window.label == "tomorrow"
    assert window.start == datetime(2026, 9, 3, 17, 0, tzinfo=UTC)
    assert window.end == datetime(2026, 9, 4, 17, 0, tzinfo=UTC)


def test_naive_clock_is_rejected() -> None:
    tool = CurrentTimeTool(
        timezone_name="Asia/Ho_Chi_Minh", clock=lambda: datetime(2026, 9, 3, 5, 30)
    )

    with pytest.raises(ValueError, match="aware datetime"):
        tool.get_current_datetime()
