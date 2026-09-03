"""Plaintext decision functions that run only on the trusted client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from voiceauth.config import EMBEDDING_DIMENSION
from voiceauth.errors import EnrollmentError


@dataclass(frozen=True)
class CandidateScore:
    identity_id: str
    display_name: str
    score: float


@dataclass(frozen=True)
class IdentificationResult:
    matched: bool
    identity_id: str | None
    display_name: str | None
    score: float | None
    candidate_count: int


@dataclass(frozen=True)
class VerificationResult:
    """Result of a 1:1 speaker-verification request.

    ``target_identity_id`` is intentionally retained even when verification
    fails.  This makes it impossible for transport code to mistake a 1:N
    identification result for an authorization decision.
    """

    matched: bool
    identity_id: str | None
    display_name: str | None
    score: float | None
    candidate_count: int
    target_identity_id: str


def normalize_embedding(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (EMBEDDING_DIMENSION,) or not np.isfinite(vector).all():
        raise EnrollmentError("The speaker model returned an invalid embedding.")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not np.isfinite(norm):
        raise EnrollmentError("The speaker embedding has an invalid norm.")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def average_enrollment(embeddings: Iterable[np.ndarray]) -> np.ndarray:
    samples = list(embeddings)
    if len(samples) != 3:
        raise EnrollmentError("Enrollment requires exactly three voice recordings.")
    return normalize_embedding(np.stack([normalize_embedding(item) for item in samples]).mean(axis=0))


def cosine_from_squared_distance(distance: float) -> float:
    """For two normalized vectors, cosine = 1 - ||a-b||² / 2."""
    if not np.isfinite(distance) or distance < -0.02:
        raise EnrollmentError("The matcher returned an invalid encrypted distance.")
    return float(np.clip(1.0 - max(distance, 0.0) / 2.0, -1.0, 1.0))


def identify(candidates: Iterable[CandidateScore], *, threshold: float) -> IdentificationResult:
    materialized = list(candidates)
    valid = [candidate for candidate in materialized if candidate.score >= threshold]
    if not valid:
        return IdentificationResult(False, None, None, None, len(materialized))
    winner = max(valid, key=lambda candidate: candidate.score)
    return IdentificationResult(True, winner.identity_id, winner.display_name, winner.score, len(materialized))


def verify(
    candidates: Iterable[CandidateScore], *, threshold: float, target_identity_id: str
) -> VerificationResult:
    """Make a 1:1 decision from the matcher response for one known identity."""

    result = identify(candidates, threshold=threshold)
    return VerificationResult(
        matched=result.matched and result.identity_id == target_identity_id,
        identity_id=result.identity_id if result.matched else None,
        display_name=result.display_name if result.matched else None,
        score=result.score,
        candidate_count=result.candidate_count,
        target_identity_id=target_identity_id,
    )
