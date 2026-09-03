"""Utterance-level embedding aggregation, caching, and trial scoring."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from speaker_recognition.evaluation.trials import (
    VerificationTrial,
    trial_list_sha256,
)


class EmbeddingEvaluationError(ValueError):
    """Raised when embeddings cannot support reliable trial scoring."""


@dataclass(frozen=True, slots=True)
class EmbeddingTable:
    """Cache exactly one normalized embedding for every utterance."""

    utterance_ids: tuple[str, ...]
    embeddings: NDArray[np.float32]

    def __post_init__(self) -> None:
        """Require unique IDs and finite L2-normalized float32 rows."""
        if not self.utterance_ids:
            raise EmbeddingEvaluationError("Embedding table must not be empty.")
        if len(set(self.utterance_ids)) != len(self.utterance_ids):
            raise EmbeddingEvaluationError(
                "Embedding table utterance IDs must be unique."
            )
        if any(
            not isinstance(utterance_id, str) or not utterance_id.strip()
            for utterance_id in self.utterance_ids
        ):
            raise EmbeddingEvaluationError(
                "Embedding table IDs must be non-empty strings."
            )
        if (
            not isinstance(self.embeddings, np.ndarray)
            or self.embeddings.dtype != np.float32
            or self.embeddings.ndim != 2
            or self.embeddings.shape[0] != len(self.utterance_ids)
            or self.embeddings.shape[1] == 0
        ):
            raise EmbeddingEvaluationError(
                "Embeddings must be float32 [utterance, dimension]."
            )
        if not self.embeddings.flags.c_contiguous:
            raise EmbeddingEvaluationError("Embeddings must be C-contiguous.")
        if not np.isfinite(self.embeddings).all():
            raise EmbeddingEvaluationError("Embeddings must all be finite.")
        norms = np.linalg.norm(self.embeddings, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
            raise EmbeddingEvaluationError(
                "Every utterance embedding must be L2-normalized."
            )
        # Detach the cache from caller-owned memory and make accidental metric
        # drift impossible after its fingerprint has been recorded.
        immutable = np.array(self.embeddings, dtype=np.float32, order="C")
        immutable.setflags(write=False)
        object.__setattr__(self, "embeddings", immutable)

    @property
    def embedding_dim(self) -> int:
        """Return the shared embedding width."""
        return int(self.embeddings.shape[1])

    @property
    def sha256(self) -> str:
        """Fingerprint ordered IDs, shape, dtype, and embedding bytes."""
        digest = hashlib.sha256()
        digest.update(b"speaker-embedding-table-v1\n")
        digest.update(str(self.embeddings.dtype).encode("ascii") + b"\n")
        digest.update(
            f"{self.embeddings.shape[0]}x{self.embeddings.shape[1]}\n".encode(
                "ascii"
            )
        )
        for utterance_id, embedding in zip(
            self.utterance_ids,
            self.embeddings,
            strict=True,
        ):
            digest.update(utterance_id.encode("utf-8") + b"\0")
            digest.update(embedding.tobytes(order="C"))
            digest.update(b"\n")
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ScoredTrialSet:
    """Store labels and cosine scores aligned to one immutable trial list."""

    labels: NDArray[np.int8]
    scores: NDArray[np.float64]
    trial_list_sha256: str

    def __post_init__(self) -> None:
        """Validate arrays before metric computation."""
        if (
            self.labels.dtype != np.int8
            or self.scores.dtype != np.float64
            or self.labels.ndim != 1
            or self.scores.ndim != 1
            or self.labels.size == 0
            or self.labels.size != self.scores.size
        ):
            raise EmbeddingEvaluationError(
                "Scored trial labels/scores must be aligned non-empty vectors."
            )
        if not np.isin(self.labels, (0, 1)).all():
            raise EmbeddingEvaluationError("Trial labels must be binary.")
        if not np.isfinite(self.scores).all():
            raise EmbeddingEvaluationError("Trial scores must be finite.")
        _validate_sha256(self.trial_list_sha256, "trial_list_sha256")
        immutable_labels = np.array(self.labels, dtype=np.int8, order="C")
        immutable_scores = np.array(self.scores, dtype=np.float64, order="C")
        immutable_labels.setflags(write=False)
        immutable_scores.setflags(write=False)
        object.__setattr__(self, "labels", immutable_labels)
        object.__setattr__(self, "scores", immutable_scores)


def aggregate_crop_embeddings(
    crop_embeddings: NDArray[np.floating],
    segment_offsets: NDArray[np.integer],
) -> NDArray[np.float32]:
    """Mean multiple crop vectors per utterance and L2-normalize the result."""
    values = np.asarray(crop_embeddings)
    offsets = np.asarray(segment_offsets)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise EmbeddingEvaluationError(
            "crop_embeddings must have non-empty shape [crop, dimension]."
        )
    if not np.isfinite(values).all():
        raise EmbeddingEvaluationError("Crop embeddings must be finite.")
    if (
        offsets.ndim != 1
        or offsets.size < 2
        or offsets[0] != 0
        or offsets[-1] != values.shape[0]
        or np.any(np.diff(offsets) <= 0)
    ):
        raise EmbeddingEvaluationError(
            "segment_offsets must partition every crop into non-empty groups."
        )

    aggregated = np.empty(
        (offsets.size - 1, values.shape[1]),
        dtype=np.float32,
    )
    for utterance_index, (start, end) in enumerate(
        zip(offsets[:-1], offsets[1:], strict=True)
    ):
        aggregated[utterance_index] = values[int(start) : int(end)].mean(
            axis=0,
            dtype=np.float64,
        )
    norms = np.linalg.norm(aggregated, axis=1, keepdims=True)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise EmbeddingEvaluationError(
            "Aggregated embeddings must have finite non-zero norm."
        )
    aggregated /= norms
    return np.ascontiguousarray(aggregated, dtype=np.float32)


def score_verification_trials(
    table: EmbeddingTable,
    trials: Sequence[VerificationTrial],
) -> ScoredTrialSet:
    """Score each trial from cached utterance embeddings exactly once."""
    snapshot = tuple(trials)
    if not snapshot:
        raise EmbeddingEvaluationError("Trial list must not be empty.")
    _validate_trials(snapshot)
    index_by_id = {
        utterance_id: index
        for index, utterance_id in enumerate(table.utterance_ids)
    }
    missing_ids = sorted(
        {
            utterance_id
            for trial in snapshot
            for utterance_id in (
                trial.left_utterance_id,
                trial.right_utterance_id,
            )
            if utterance_id not in index_by_id
        }
    )
    if missing_ids:
        examples = missing_ids[:5]
        raise EmbeddingEvaluationError(
            "Trial utterances are absent from the embedding cache: "
            f"{examples}."
        )

    left_indices = np.fromiter(
        (index_by_id[trial.left_utterance_id] for trial in snapshot),
        dtype=np.int64,
        count=len(snapshot),
    )
    right_indices = np.fromiter(
        (index_by_id[trial.right_utterance_id] for trial in snapshot),
        dtype=np.int64,
        count=len(snapshot),
    )
    labels = np.fromiter(
        (trial.label for trial in snapshot),
        dtype=np.int8,
        count=len(snapshot),
    )
    scores = np.einsum(
        "ij,ij->i",
        table.embeddings[left_indices],
        table.embeddings[right_indices],
        dtype=np.float64,
        optimize=True,
    )
    scores = np.clip(scores, -1.0, 1.0)
    return ScoredTrialSet(
        labels=np.ascontiguousarray(labels, dtype=np.int8),
        scores=np.ascontiguousarray(scores, dtype=np.float64),
        trial_list_sha256=trial_list_sha256(snapshot),
    )


def _validate_sha256(value: object, field_name: str) -> None:
    """Require a lowercase hexadecimal SHA-256 digest."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EmbeddingEvaluationError(
            f"{field_name} must be a lowercase SHA-256 digest."
        )


def _validate_trials(trials: tuple[VerificationTrial, ...]) -> None:
    """Reject duplicate or semantically inconsistent verification trials."""
    trial_ids: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for trial in trials:
        if not trial.trial_id.strip() or trial.trial_id in trial_ids:
            raise EmbeddingEvaluationError(
                "Trial IDs must be non-empty and unique."
            )
        trial_ids.add(trial.trial_id)
        if trial.left_utterance_id == trial.right_utterance_id:
            raise EmbeddingEvaluationError(
                "A verification trial must contain two distinct utterances."
            )
        pair = tuple(
            sorted((trial.left_utterance_id, trial.right_utterance_id))
        )
        if pair in pairs:
            raise EmbeddingEvaluationError(
                "Verification trial utterance pairs must be unique."
            )
        pairs.add(pair)
        speakers_match = trial.left_speaker_id == trial.right_speaker_id
        if trial.label not in {0, 1} or speakers_match != bool(trial.label):
            raise EmbeddingEvaluationError(
                "Trial labels must agree with same/different speaker IDs."
            )
