"""Trusted construction of the persistent local voice-auth dependency graph."""

from __future__ import annotations

from pathlib import Path

from voiceauth.config import VoiceSettings
from voiceauth.gate import VoiceAuthGate
from voiceauth.session_factory import VoiceAuthSessionFactory


def build_persistent_voice_auth_gate() -> VoiceAuthGate:
    # `.env.local` model paths are documented relative to this app directory,
    # while the Agent itself is launched from the workspace root.
    voice_app_directory = Path(__file__).resolve().parents[1] / "voice_verification"
    settings = VoiceSettings.from_environment(root=voice_app_directory)
    service = VoiceAuthSessionFactory(settings).create(persistent_context=True)
    return VoiceAuthGate(service)


def build_account_voice_auth_gate(account_id: str) -> VoiceAuthGate:
    """Build an account-scoped local HE service for the gateway or Agent.

    A context is never shared between accounts. The account ID is supplied by
    a trusted backend session or signed LiveKit dispatch metadata, never by an
    LLM or transcript.
    """
    if not account_id.strip():
        raise ValueError("A trusted account ID is required for voice authentication.")
    voice_app_directory = Path(__file__).resolve().parents[1] / "voice_verification"
    settings = VoiceSettings.from_environment(root=voice_app_directory)
    service = VoiceAuthSessionFactory(settings).create(
        persistent_context=True,
        keychain_account=f"{settings.keychain_account}:{account_id}",
    )
    return VoiceAuthGate(service)
