"""Tests for shared speaker-verification metrics."""

import numpy as np
import pytest

from speaker_recognition.evaluation.metrics import (
    VerificationMetricError,
    build_roc_curve,
    compute_eer,
    compute_min_dcf,
    compute_operating_point,
    compute_verification_metrics,
    select_tar_at_far,
)


def test_perfect_separation_has_zero_eer_and_min_dcf() -> None:
    """A perfectly ordered score list should achieve zero detection error."""
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    curve = build_roc_curve(labels, scores)

    eer, _ = compute_eer(curve)
    min_dcf = compute_min_dcf(curve)
    point = compute_operating_point(labels, scores, threshold=0.5)

    assert eer == pytest.approx(0.0)
    assert min_dcf.value == pytest.approx(0.0)
    assert point.far == pytest.approx(0.0)
    assert point.frr == pytest.approx(0.0)
    assert point.accuracy == pytest.approx(1.0)


def test_operating_point_reports_known_confusion_matrix() -> None:
    """FAR, FRR, TAR, and accuracy should follow explicit counts."""
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.4, 0.6, 0.1])

    point = compute_operating_point(labels, scores, threshold=0.5)

    assert point.true_accepts == 1
    assert point.false_rejects == 1
    assert point.false_accepts == 1
    assert point.true_rejects == 1
    assert point.far == pytest.approx(0.5)
    assert point.frr == pytest.approx(0.5)
    assert point.tar == pytest.approx(0.5)
    assert point.accuracy == pytest.approx(0.5)


def test_all_tied_scores_use_interpolated_half_eer() -> None:
    """Tied scores must not create impossible partial acceptance points."""
    curve = build_roc_curve(
        np.array([1, 0]),
        np.array([0.5, 0.5]),
    )

    eer, threshold = compute_eer(curve)

    assert curve.thresholds.size == 2
    assert eer == pytest.approx(0.5)
    assert threshold == pytest.approx(0.5)


def test_tar_at_far_selects_best_feasible_empirical_point() -> None:
    """TAR selection must never exceed the requested empirical FAR."""
    labels = np.array([1, 1, 1, 0, 0, 0, 0])
    scores = np.array([0.95, 0.8, 0.3, 0.9, 0.4, 0.2, 0.1])
    curve = build_roc_curve(labels, scores)

    point = select_tar_at_far(curve, far_target=0.25)

    assert point.far <= 0.25
    assert point.tar == pytest.approx(2 / 3)


def test_complete_metric_names_include_required_far_targets() -> None:
    """Flat output should expose every required reporting metric."""
    metrics = compute_verification_metrics(
        np.array([1, 1, 0, 0]),
        np.array([0.9, 0.8, 0.2, 0.1]),
        decision_threshold=0.5,
    )

    flattened = metrics.to_flat_dict()

    assert flattened["eer"] == pytest.approx(0.0)
    assert flattened["accuracy"] == pytest.approx(1.0)
    assert flattened["tar_at_far_5pct"] == pytest.approx(1.0)
    assert flattened["tar_at_far_1pct"] == pytest.approx(1.0)
    assert flattened["tar_at_far_0p1pct"] == pytest.approx(1.0)
    assert flattened["tar_at_far_0p01pct"] == pytest.approx(1.0)


def test_metrics_are_invariant_to_trial_input_order() -> None:
    """Trial list ordering must not alter any scalar result."""
    labels = np.array([1, 0, 1, 0, 1, 0])
    scores = np.array([0.8, 0.3, 0.7, 0.6, 0.4, 0.2])
    permutation = np.array([5, 2, 0, 4, 1, 3])

    first = compute_verification_metrics(
        labels,
        scores,
        decision_threshold=0.5,
    ).to_flat_dict()
    reordered = compute_verification_metrics(
        labels[permutation],
        scores[permutation],
        decision_threshold=0.5,
    ).to_flat_dict()

    assert reordered == first


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        ([1, 1], [0.8, 0.7]),
        ([0, 0], [0.2, 0.1]),
        ([1, 0], [0.5, np.nan]),
        ([1, 2], [0.5, 0.4]),
        ([1], [0.5, 0.4]),
    ],
)
def test_reject_invalid_trial_arrays(
    labels: list[int],
    scores: list[float],
) -> None:
    """Metrics require aligned finite binary trials with both classes."""
    with pytest.raises(VerificationMetricError):
        build_roc_curve(labels, scores)
