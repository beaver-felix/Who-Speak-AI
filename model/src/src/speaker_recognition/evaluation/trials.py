"""Deterministic speaker-balanced verification trial construction."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from speaker_recognition.data.manifest import ManifestRecord, Split


class TrialConstructionError(ValueError):
    """Raised when records cannot support the requested verification trials."""


@dataclass(frozen=True, slots=True)
class VerificationTrial:
    """Identify one unordered genuine or impostor utterance pair."""

    trial_id: str
    label: int
    left_utterance_id: str
    right_utterance_id: str
    left_speaker_id: str
    right_speaker_id: str


def build_verification_trials(
    records: tuple[ManifestRecord, ...] | list[ManifestRecord],
    *,
    split: Split,
    seed: int,
    max_genuine_per_speaker: int = 20,
    impostor_trial_count: int = 100_000,
) -> tuple[VerificationTrial, ...]:
    """Build deterministic genuine and speaker-balanced impostor trials.

    Genuine trials are capped per speaker so speakers with many recordings do
    not dominate. Impostor generation samples speaker identities uniformly
    before sampling one utterance from each identity, reducing the influence of
    high-resource speakers.

    Parameters
    ----------
    records:
        Canonical manifest rows. Only rows in ``split`` are used.
    split:
        Canonical Validation or Test partition.
    seed:
        Non-negative deterministic random seed.
    max_genuine_per_speaker:
        Maximum unique same-speaker pairs retained per speaker.
    impostor_trial_count:
        Exact number of unique different-speaker pairs requested.

    Returns
    -------
    tuple[VerificationTrial, ...]
        Genuine trials first, followed by impostor trials; each section is
        sorted by stable utterance identifiers.

    Raises
    ------
    TrialConstructionError
        If settings, identities, or unique-pair capacity are insufficient.
    """
    if split not in {Split.VALIDATION, Split.TEST}:
        raise TrialConstructionError(
            "Verification trials require validation or test records."
        )
    if seed < 0:
        raise TrialConstructionError("seed must be non-negative.")
    if max_genuine_per_speaker <= 0:
        raise TrialConstructionError(
            "max_genuine_per_speaker must be positive."
        )
    if impostor_trial_count <= 0:
        raise TrialConstructionError("impostor_trial_count must be positive.")

    selected_records = [record for record in records if record.split is split]
    if not selected_records:
        raise TrialConstructionError(f"No records found for split {split.value!r}.")

    records_by_speaker: dict[str, list[ManifestRecord]] = defaultdict(list)
    seen_utterance_ids: set[str] = set()
    for record in selected_records:
        if record.utterance_id in seen_utterance_ids:
            raise TrialConstructionError(
                f"Duplicate utterance ID: {record.utterance_id!r}."
            )
        seen_utterance_ids.add(record.utterance_id)
        records_by_speaker[record.speaker_id].append(record)

    if len(records_by_speaker) < 2:
        raise TrialConstructionError(
            "At least two speakers are required for impostor trials."
        )

    for speaker_records in records_by_speaker.values():
        speaker_records.sort(key=lambda record: record.utterance_id)

    generator = np.random.default_rng(seed)
    genuine_pairs: list[tuple[ManifestRecord, ManifestRecord]] = []
    for speaker_id in sorted(records_by_speaker):
        speaker_records = records_by_speaker[speaker_id]
        pair_capacity = len(speaker_records) * (len(speaker_records) - 1) // 2
        if pair_capacity == 0:
            continue
        target_count = min(pair_capacity, max_genuine_per_speaker)
        pairs = _sample_genuine_pairs(
            speaker_records,
            target_count=target_count,
            generator=generator,
        )
        genuine_pairs.extend(pairs)

    if not genuine_pairs:
        raise TrialConstructionError(
            "At least one speaker with two utterances is required."
        )

    utterance_count = len(selected_records)
    all_pair_capacity = utterance_count * (utterance_count - 1) // 2
    genuine_pair_capacity = sum(
        len(speaker_records) * (len(speaker_records) - 1) // 2
        for speaker_records in records_by_speaker.values()
    )
    impostor_pair_capacity = all_pair_capacity - genuine_pair_capacity
    if impostor_trial_count > impostor_pair_capacity:
        raise TrialConstructionError(
            f"Requested {impostor_trial_count:,} impostor trials, but only "
            f"{impostor_pair_capacity:,} unique pairs exist."
        )

    impostor_pairs = _sample_impostor_pairs(
        records_by_speaker,
        target_count=impostor_trial_count,
        generator=generator,
    )

    genuine_trials = sorted(
        (_make_trial(left, right, label=1) for left, right in genuine_pairs),
        key=lambda trial: trial.trial_id,
    )
    impostor_trials = sorted(
        (_make_trial(left, right, label=0) for left, right in impostor_pairs),
        key=lambda trial: trial.trial_id,
    )
    return tuple(genuine_trials + impostor_trials)


def trial_list_sha256(trials: tuple[VerificationTrial, ...]) -> str:
    """Fingerprint a complete ordered trial protocol."""
    if not trials:
        raise TrialConstructionError("Cannot hash an empty trial list.")

    digest = hashlib.sha256()
    for trial in trials:
        line = "\t".join(
            (
                trial.trial_id,
                str(trial.label),
                trial.left_utterance_id,
                trial.right_utterance_id,
                trial.left_speaker_id,
                trial.right_speaker_id,
            )
        )
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sample_genuine_pairs(
    records: list[ManifestRecord],
    *,
    target_count: int,
    generator: np.random.Generator,
) -> list[tuple[ManifestRecord, ManifestRecord]]:
    """Sample unique same-speaker pairs without enumerating large pair sets."""
    pair_capacity = len(records) * (len(records) - 1) // 2
    if target_count == pair_capacity:
        return list(combinations(records, 2))

    selected_indices: set[tuple[int, int]] = set()
    while len(selected_indices) < target_count:
        indices = generator.choice(len(records), size=2, replace=False)
        left_index, right_index = sorted(int(index) for index in indices)
        selected_indices.add((left_index, right_index))

    return [
        (records[left_index], records[right_index])
        for left_index, right_index in sorted(selected_indices)
    ]


def _sample_impostor_pairs(
    records_by_speaker: dict[str, list[ManifestRecord]],
    *,
    target_count: int,
    generator: np.random.Generator,
) -> list[tuple[ManifestRecord, ManifestRecord]]:
    """Sample unique pairs after choosing speaker identities uniformly."""
    speaker_ids = tuple(sorted(records_by_speaker))
    selected_pairs: dict[
        tuple[str, str],
        tuple[ManifestRecord, ManifestRecord],
    ] = {}
    maximum_attempts = target_count * 100 + 10_000
    attempts = 0

    while len(selected_pairs) < target_count and attempts < maximum_attempts:
        speaker_indices = generator.choice(
            len(speaker_ids),
            size=2,
            replace=False,
        )
        first_speaker, second_speaker = sorted(
            speaker_ids[int(index)] for index in speaker_indices
        )
        first_records = records_by_speaker[first_speaker]
        second_records = records_by_speaker[second_speaker]
        first_record = first_records[
            int(generator.integers(0, len(first_records)))
        ]
        second_record = second_records[
            int(generator.integers(0, len(second_records)))
        ]
        key = (
            first_record.utterance_id,
            second_record.utterance_id,
        )
        selected_pairs[key] = (first_record, second_record)
        attempts += 1

    if len(selected_pairs) != target_count:
        raise TrialConstructionError(
            "Unable to sample the requested unique impostor trials within "
            "the deterministic attempt limit."
        )
    return [selected_pairs[key] for key in sorted(selected_pairs)]


def _make_trial(
    left: ManifestRecord,
    right: ManifestRecord,
    *,
    label: int,
) -> VerificationTrial:
    """Create one canonically ordered trial and stable identifier."""
    if right.utterance_id < left.utterance_id:
        left, right = right, left
    prefix = "genuine" if label == 1 else "impostor"
    trial_id = f"{prefix}:{left.utterance_id}|{right.utterance_id}"
    return VerificationTrial(
        trial_id=trial_id,
        label=label,
        left_utterance_id=left.utterance_id,
        right_utterance_id=right.utterance_id,
        left_speaker_id=left.speaker_id,
        right_speaker_id=right.speaker_id,
    )
