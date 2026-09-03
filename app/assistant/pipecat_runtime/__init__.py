"""Pipecat runtime for the local-first Who Speak AI Agent.

The package intentionally contains only the application adapters. Pipecat
itself remains a pinned dependency and the legacy LiveKit Agents runtime stays
available as a fallback while this runtime is validated.
"""

from app.assistant.pipecat_runtime.session import (
    PipecatSessionDescriptor,
    sign_session_descriptor,
    verify_session_descriptor,
)

__all__ = [
    "PipecatSessionDescriptor",
    "sign_session_descriptor",
    "verify_session_descriptor",
]
