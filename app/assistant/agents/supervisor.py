"""Permission boundary for the future Supervisor Agent."""

from __future__ import annotations

from app.assistant.contracts import AuthDecision


class SupervisorPolicy:
    """Select only tools allowed for the authentication state."""

    PUBLIC_TOOLS = frozenset({"general_qa"})
    PRIVATE_TOOLS = frozenset({"calendar.list_events", "calendar.create_event"})

    @staticmethod
    def allowed_tool_names(auth: AuthDecision, *, public_tools: set[str], private_tools: set[str]) -> set[str]:
        return public_tools | private_tools if auth.may_use_private_tools else set(public_tools)

    @classmethod
    def for_auth(cls, auth: AuthDecision) -> set[str]:
        return cls.allowed_tool_names(auth, public_tools=set(cls.PUBLIC_TOOLS), private_tools=set(cls.PRIVATE_TOOLS))
