"""Deterministic speaker-capped record selection for training epochs."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Sequence

from speaker_recognition.data.manifest import ManifestRecord, Split


class EpochSamplingError(ValueError):
    """Raised when records cannot form a deterministic training epoch."""


def select_speaker_capped_epoch(
    records: Sequence[ManifestRecord],
    *,
    max_utterances_per_speaker: int,
    max_speakers: int | None = None,
    seed: int,
    epoch_index: int,
) -> tuple[ManifestRecord, ...]:
    """Select up to a fixed number of utterances from every train speaker.

    Selection ranks each utterance by a SHA-256 value derived from the run
    seed, epoch, speaker, and utterance ID. It therefore changes exposure over
    epochs without consuming global random state and can be reconstructed
    exactly after a checkpoint resume.
    """
    for value, field_name, positive in (
        (max_utterances_per_speaker, "max_utterances_per_speaker", True),
        (seed, "seed", False),
        (epoch_index, "epoch_index", False),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or (value <= 0 if positive else value < 0)
        ):
            qualifier = "positive" if positive else "non-negative"
            raise EpochSamplingError(
                f"{field_name} must be a {qualifier} integer."
            )
    if max_speakers is not None and (
        isinstance(max_speakers, bool)
        or not isinstance(max_speakers, int)
        or max_speakers <= 0
    ):
        raise EpochSamplingError(
            "max_speakers must be a positive integer or None."
        )

    records_by_speaker: dict[str, list[ManifestRecord]] = defaultdict(list)
    seen_utterances: set[str] = set()
    for record in records:
        if record.split is not Split.TRAIN:
            raise EpochSamplingError(
                "Speaker-capped epoch selection accepts Train records only."
            )
        if record.utterance_id in seen_utterances:
            raise EpochSamplingError(
                f"Duplicate utterance ID: {record.utterance_id!r}."
            )
        seen_utterances.add(record.utterance_id)
        records_by_speaker[record.speaker_id].append(record)
    if not records_by_speaker:
        raise EpochSamplingError("Training epoch records must not be empty.")

    speaker_ids = sorted(records_by_speaker)
    if max_speakers is not None:
        # Pilot membership stays bounded while rotating deterministically if a
        # later pilot epoch is ever requested. The full stage passes None and
        # therefore retains every canonical Train speaker.
        speaker_ids = sorted(
            speaker_ids,
            key=lambda speaker_id: (
                _speaker_selection_digest(
                    seed=seed,
                    epoch_index=epoch_index,
                    speaker_id=speaker_id,
                ),
                speaker_id,
            ),
        )[:max_speakers]

    selected: list[ManifestRecord] = []
    for speaker_id in speaker_ids:
        ranked = sorted(
            records_by_speaker[speaker_id],
            key=lambda record: (
                _selection_digest(
                    seed=seed,
                    epoch_index=epoch_index,
                    speaker_id=speaker_id,
                    utterance_id=record.utterance_id,
                ),
                record.utterance_id,
            ),
        )
        selected.extend(ranked[:max_utterances_per_speaker])

    # The engine owns the final epoch shuffle. A lexical base order makes the
    # selected membership independently inspectable and input-order invariant.
    return tuple(sorted(selected, key=lambda record: record.utterance_id))


def utterance_id_sha256(records: Sequence[ManifestRecord]) -> str:
    """Fingerprint one ordered epoch membership without physical audio reads."""
    snapshot = tuple(records)
    if not snapshot:
        raise EpochSamplingError("Cannot hash an empty epoch membership.")
    digest = hashlib.sha256()
    digest.update(b"speaker-capped-epoch-membership-v1\n")
    for record in snapshot:
        digest.update(record.utterance_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _selection_digest(
    *,
    seed: int,
    epoch_index: int,
    speaker_id: str,
    utterance_id: str,
) -> bytes:
    """Return the stable private ranking key for one utterance."""
    payload = (
        f"speaker-capped-epoch-v1\0{seed}\0{epoch_index}\0"
        f"{speaker_id}\0{utterance_id}"
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _speaker_selection_digest(
    *,
    seed: int,
    epoch_index: int,
    speaker_id: str,
) -> bytes:
    """Return a stable private ranking key for one pilot speaker."""
    payload = (
        f"speaker-capped-epoch-speaker-v1\0{seed}\0{epoch_index}\0"
        f"{speaker_id}"
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()
