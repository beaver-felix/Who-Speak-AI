from datetime import UTC, datetime, timedelta

from app.assistant.contracts import AuthDecision, AuthState
from app.assistant.agents.supervisor import SupervisorPolicy


def test_guest_cannot_access_private_tools() -> None:
    allowed = SupervisorPolicy.allowed_tool_names(
        AuthDecision(AuthState.GUEST),
        public_tools={"weather"},
        private_tools={"read_diary"},
    )
    assert allowed == {"weather"}


def test_authenticated_identity_can_access_private_tools() -> None:
    allowed = SupervisorPolicy.allowed_tool_names(
        AuthDecision(
            AuthState.AUTHENTICATED,
            "identity-1",
            "Thanh",
            datetime.now(UTC) + timedelta(minutes=5),
        ),
        public_tools={"weather"},
        private_tools={"read_diary"},
    )
    assert allowed == {"weather", "read_diary"}


def test_expired_identity_cannot_access_private_tools() -> None:
    allowed = SupervisorPolicy.allowed_tool_names(
        AuthDecision(
            AuthState.AUTHENTICATED,
            "identity-1",
            "Thanh",
            datetime.now(UTC) - timedelta(seconds=1),
        ),
        public_tools={"weather"},
        private_tools={"read_diary"},
    )
    assert allowed == {"weather"}
