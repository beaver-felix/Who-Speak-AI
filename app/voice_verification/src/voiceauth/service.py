"""Trusted-client enrollment and identification workflow."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

from voiceauth.audio import AudioRecording, validate_speech_sample
from voiceauth.he import VoiceHEContext
from voiceauth.matcher_client import MatcherClient
from voiceauth.matching import (
    CandidateScore,
    IdentificationResult,
    VerificationResult,
    average_enrollment,
    cosine_from_squared_distance,
    identify,
    verify,
)


@dataclass(frozen=True)
class EnrollmentResult:
    identity_id: str
    display_name: str


class VoiceVerificationService:
    def __init__(self, engine, matcher: MatcherClient, context: VoiceHEContext, *, threshold: float) -> None:
        self.engine = engine
        self.matcher = matcher
        self.context = context
        self.threshold = threshold

    def initialize(self) -> None:
        self.engine.prepare()
        self.matcher.register_context(self.context)

    def enroll(self, display_name: str, recordings: list[AudioRecording]) -> EnrollmentResult:
        name = display_name.strip()
        if not name:
            raise ValueError("Enter a display name before enrolling.")
        for recording in recordings:
            validate_speech_sample(recording)
        embeddings = [self.engine.embed(recording.waveform, recording.sample_rate) for recording in recordings]
        template = average_enrollment(embeddings)
        encrypted = self.context.client.encrypt_embedding(template)
        identity_id = uuid4()
        self.matcher.enroll_identity(self.context.context_id, identity_id, name, encrypted)
        return EnrollmentResult(str(identity_id), name)

    def identify(self, recording: AudioRecording, *, target_identity_id: UUID | None = None) -> IdentificationResult:
        validate_speech_sample(recording)
        query = self.engine.embed(recording.waveform, recording.sample_rate)
        encrypted = self.context.client.encrypt_embedding(query)
        comparisons = self.matcher.match(self.context.context_id, encrypted, target_identity_id)
        scores = [
            CandidateScore(
                identity_id=candidate.identity_id,
                display_name=candidate.display_name,
                score=cosine_from_squared_distance(self.context.client.decrypt_distance(candidate.encrypted_distance_b64)),
            )
            for candidate in comparisons
        ]
        return identify(scores, threshold=self.threshold)

    def verify(self, recording: AudioRecording, *, target_identity_id: UUID) -> VerificationResult:
        """Verify only the requested identity; this is never a 1:N operation."""

        result = self.identify(recording, target_identity_id=target_identity_id)
        return VerificationResult(
            matched=result.matched and result.identity_id == str(target_identity_id),
            identity_id=result.identity_id if result.matched else None,
            display_name=result.display_name if result.matched else None,
            score=result.score,
            candidate_count=result.candidate_count,
            target_identity_id=str(target_identity_id),
        )
