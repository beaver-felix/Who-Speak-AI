"""Stable speaker-authentication interface for Streamlit and future LiveKit."""

from __future__ import annotations

from uuid import UUID

from voiceauth.audio import AudioRecording
from voiceauth.matching import IdentificationResult, VerificationResult
from voiceauth.service import EnrollmentResult, VoiceVerificationService


class VoiceAuthGate:
    """Expose identification and 1:1 verification without leaking internals.

    Transport code may construct an ``AudioRecording`` from its trusted audio
    stream, but it never receives the RawNet3 runtime, HE private context, or
    matcher client.
    """

    def __init__(self, service: VoiceVerificationService) -> None:
        self._service = service

    def identify(self, audio: AudioRecording) -> IdentificationResult:
        return self._service.identify(audio)

    def enroll(self, display_name: str, recordings: list[AudioRecording]) -> EnrollmentResult:
        """Trusted enrollment entrypoint for the local account gateway."""
        return self._service.enroll(display_name, recordings)

    @property
    def context_id(self) -> str:
        return str(self._service.context.context_id)

    def verify(self, audio: AudioRecording, *, target_identity_id: str | UUID) -> VerificationResult:
        return self._service.verify(audio, target_identity_id=UUID(str(target_identity_id)))

    def close(self) -> None:
        """Release only the matcher transport; private HE material remains local."""
        self._service.matcher.close()
