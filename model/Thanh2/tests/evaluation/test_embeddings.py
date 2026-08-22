"""Tests for crop aggregation and cache-only verification scoring."""

import numpy as np
import pytest

from speaker_recognition.evaluation.embeddings import (
    EmbeddingEvaluationError,
    EmbeddingTable,
    aggregate_crop_embeddings,
    score_verification_trials,
)
from speaker_recognition.evaluation.trials import (
    VerificationTrial,
    trial_list_sha256,
)


def _trial(
    trial_id: str,
    label: int,
    left: str,
    right: str,
) -> VerificationTrial:
    """Create a compact test trial with consistent speaker labels."""
    return VerificationTrial(
        trial_id=trial_id,
        label=label,
        left_utterance_id=left,
        right_utterance_id=right,
        left_speaker_id="speaker-a",
        right_speaker_id="speaker-a" if label else "speaker-b",
    )


def test_crop_aggregation_means_each_group_then_normalizes() -> None:
    """Crop count must not change the scale of an utterance embedding."""
    crop_embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-2.0, 0.0]],
        dtype=np.float32,
    )

    aggregated = aggregate_crop_embeddings(
        crop_embeddings,
        np.array([0, 2, 3], dtype=np.int64),
    )

    expected = np.array(
        [[2**-0.5, 2**-0.5], [-1.0, 0.0]],
        dtype=np.float32,
    )
    assert aggregated.dtype == np.float32
    assert aggregated.flags.c_contiguous
    assert aggregated == pytest.approx(expected)


@pytest.mark.parametrize(
    "offsets",
    [
        np.array([1, 3]),
        np.array([0, 2]),
        np.array([0, 0, 3]),
        np.array([[0, 3]]),
    ],
)
def test_crop_aggregation_rejects_invalid_partitions(offsets: np.ndarray) -> None:
    """Offsets must cover every crop exactly once in non-empty groups."""
    with pytest.raises(EmbeddingEvaluationError, match="segment_offsets"):
        aggregate_crop_embeddings(
            np.ones((3, 2), dtype=np.float32),
            offsets,
        )


def test_embedding_table_requires_normalized_unique_rows() -> None:
    """A cache must reject duplicates and unnormalized vectors."""
    with pytest.raises(EmbeddingEvaluationError, match="unique"):
        EmbeddingTable(
            ("u1", "u1"),
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
    with pytest.raises(EmbeddingEvaluationError, match="L2-normalized"):
        EmbeddingTable(
            ("u1",),
            np.array([[2.0, 0.0]], dtype=np.float32),
        )


def test_trial_scoring_uses_cached_cosine_and_fingerprints_protocol() -> None:
    """One cache should score repeated trial references without re-encoding."""
    table = EmbeddingTable(
        ("u1", "u2", "u3", "u4"),
        np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        ),
    )
    trials = (
        _trial("t1", 1, "u1", "u2"),
        _trial("t2", 0, "u1", "u3"),
        _trial("t3", 0, "u1", "u4"),
    )

    scored = score_verification_trials(table, trials)

    assert scored.labels.tolist() == [1, 0, 0]
    assert scored.scores.tolist() == pytest.approx([1.0, 0.0, -1.0])
    assert scored.trial_list_sha256 == trial_list_sha256(trials)
    assert len(table.sha256) == 64
    assert not table.embeddings.flags.writeable
    assert not scored.labels.flags.writeable
    assert not scored.scores.flags.writeable


def test_trial_scoring_fails_when_cache_coverage_is_incomplete() -> None:
    """Missing trial audio must fail instead of silently reducing evaluation."""
    table = EmbeddingTable(
        ("u1",),
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(EmbeddingEvaluationError, match="absent"):
        score_verification_trials(table, (_trial("t1", 0, "u1", "u2"),))


def test_trial_scoring_rejects_semantically_invalid_protocol() -> None:
    """Labels must agree with speaker identities before any score is trusted."""
    table = EmbeddingTable(
        ("u1", "u2"),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    invalid = VerificationTrial("t1", 1, "u1", "u2", "a", "b")

    with pytest.raises(EmbeddingEvaluationError, match="agree"):
        score_verification_trials(table, (invalid,))
