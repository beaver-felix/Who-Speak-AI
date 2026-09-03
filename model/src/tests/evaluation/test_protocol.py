"""Tests for framework-neutral cached-embedding evaluation."""

import json

import numpy as np
import pytest

from speaker_recognition.evaluation.embeddings import (
    EmbeddingEvaluationError,
    EmbeddingTable,
)
from speaker_recognition.evaluation.protocol import evaluate_embedding_table
from speaker_recognition.evaluation.trials import (
    VerificationTrial,
    trial_list_sha256,
)


def _protocol() -> tuple[EmbeddingTable, tuple[VerificationTrial, ...]]:
    """Create a perfectly separated two-speaker protocol."""
    diagonal = np.float32(2**-0.5)
    table = EmbeddingTable(
        ("a1", "a2", "b1", "b2"),
        np.array(
            [
                [1.0, 0.0],
                [diagonal, diagonal],
                [-1.0, 0.0],
                [-diagonal, -diagonal],
            ],
            dtype=np.float32,
        ),
    )
    trials = (
        VerificationTrial("g1", 1, "a1", "a2", "a", "a"),
        VerificationTrial("g2", 1, "b1", "b2", "b", "b"),
        VerificationTrial("i1", 0, "a1", "b1", "a", "b"),
        VerificationTrial("i2", 0, "a2", "b2", "a", "b"),
    )
    return table, trials


def test_validation_evaluation_uses_eer_threshold_only_for_diagnostics() -> None:
    """Validation must report metrics without claiming a security threshold."""
    table, trials = _protocol()

    result = evaluate_embedding_table(
        table,
        trials,
        expected_trial_sha256=trial_list_sha256(trials),
    )
    artifact = result.to_artifact()

    assert result.metrics.eer == pytest.approx(0.0)
    assert result.metrics.min_dcf.value == pytest.approx(0.0)
    assert result.metrics.decision.accuracy == pytest.approx(1.0)
    assert artifact["threshold_policy"]["security_threshold_selected"] is False
    assert (
        artifact["threshold_policy"]["decision_threshold_source"]
        == "validation_interpolated_eer_diagnostic"
    )
    json.dumps(artifact, allow_nan=False)


def test_test_evaluation_applies_an_explicit_frozen_threshold() -> None:
    """An external Validation threshold must be recorded as frozen."""
    table, trials = _protocol()

    result = evaluate_embedding_table(
        table,
        trials,
        expected_trial_sha256=trial_list_sha256(trials),
        decision_threshold=0.0,
    )

    assert result.metrics.decision.threshold == pytest.approx(0.0)
    assert result.decision_threshold_source == (
        "frozen_validation_security_threshold"
    )


def test_evaluation_rejects_changed_trial_protocol() -> None:
    """Metrics must never be computed under an undeclared trial list."""
    table, trials = _protocol()

    with pytest.raises(EmbeddingEvaluationError, match="fingerprint"):
        evaluate_embedding_table(
            table,
            trials,
            expected_trial_sha256="0" * 64,
        )


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), True])
def test_evaluation_rejects_unsafe_external_threshold(threshold: object) -> None:
    """Frozen thresholds must be ordinary finite real numbers."""
    table, trials = _protocol()

    with pytest.raises(EmbeddingEvaluationError, match="finite and real"):
        evaluate_embedding_table(
            table,
            trials,
            expected_trial_sha256=trial_list_sha256(trials),
            decision_threshold=threshold,
        )
