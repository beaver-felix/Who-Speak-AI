"""Speaker-verification metrics used by the RawNet3 pipeline.

The functions assume that larger scores indicate a stronger match and that
labels use ``1`` for a target/genuine trial and ``0`` for an impostor trial.
The implementation intentionally uses NumPy only, so it remains usable in a
small inference or evaluation environment without adding a metrics package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_METRICS_PATH = (
    Path(__file__).resolve().parent.parent / "results" / "rawnet3_metrics.json"
)


def _validate_inputs(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and flatten score/label arrays.

    Args:
        scores: Similarity or logit scores, where larger means more likely
            genuine.
        labels: Binary ground-truth labels; ``1`` is genuine and ``0`` is
            impostor.

    Returns:
        A pair of one-dimensional ``float64`` scores and ``int64`` labels.

    Raises:
        ValueError: If the arrays are empty, have different lengths, contain
            non-finite scores, or do not contain both trial classes.
    """
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    label_array = np.asarray(labels).reshape(-1)

    if score_array.size == 0:
        raise ValueError("scores and labels must not be empty")
    if score_array.size != label_array.size:
        raise ValueError("scores and labels must have the same length")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must contain only finite values")

    # Accept booleans and numeric 0/1 values while rejecting ambiguous labels.
    try:
        label_array = label_array.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must be binary values") from exc
    if not np.isin(label_array, (0, 1)).all():
        raise ValueError("labels must contain only 0 and 1")
    if not np.any(label_array == 1) or not np.any(label_array == 0):
        raise ValueError("both genuine and impostor trials are required")

    return score_array, label_array


def _operating_points(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return thresholds, false-accept rates, and false-reject rates.

    One operating point is retained for each unique score. At a threshold,
    trials with ``score >= threshold`` are classified as genuine.
    """
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    positives = float(np.sum(labels == 1))
    negatives = float(np.sum(labels == 0))
    true_positive = np.cumsum(sorted_labels == 1, dtype=np.float64)
    false_positive = np.cumsum(sorted_labels == 0, dtype=np.float64)
    false_reject_rate = (positives - true_positive) / positives
    false_accept_rate = false_positive / negatives

    # Keep only the last trial in each tied-score group. Include a threshold
    # above the maximum score, which rejects every trial.
    last_in_group = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    thresholds = np.r_[np.inf, sorted_scores[last_in_group]]
    false_accept_rate = np.r_[0.0, false_accept_rate[last_in_group]]
    false_reject_rate = np.r_[1.0, false_reject_rate[last_in_group]]
    return thresholds, false_accept_rate, false_reject_rate


def _eer_details(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    """Calculate EER and the closest empirical decision threshold."""
    thresholds, false_accept, false_reject = _operating_points(scores, labels)
    difference = false_reject - false_accept

    # If the two curves cross between measured points, linearly interpolate
    # the crossing. This is more stable than selecting one side arbitrarily.
    crossing_indices = np.flatnonzero(difference[:-1] * difference[1:] <= 0)
    if crossing_indices.size:
        index = int(crossing_indices[0])
        left_difference = difference[index]
        right_difference = difference[index + 1]
        denominator = left_difference - right_difference
        fraction = 0.0 if denominator == 0 else left_difference / denominator
        eer = false_accept[index] + fraction * (
            false_accept[index + 1] - false_accept[index]
        )
    else:
        index = int(np.argmin(np.abs(difference)))
        eer = (false_accept[index] + false_reject[index]) / 2.0

    closest_index = int(np.argmin(np.abs(difference)))
    threshold = float(thresholds[closest_index])
    return float(np.clip(eer, 0.0, 1.0)), threshold


def compute_eer(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> float:
    """Compute the equal error rate (EER).

    Args:
        scores: Verification scores; higher values mean more similar.
        labels: Binary labels with genuine ``1`` and impostor ``0``.

    Returns:
        EER in the range ``[0, 1]``. Multiply by 100 when displaying a
        percentage.
    """
    score_array, label_array = _validate_inputs(scores, labels)
    eer, _ = _eer_details(score_array, label_array)
    return eer


def compute_min_dcf(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    target_prior: float = 0.01,
    c_miss: float = 1.0,
    c_false_alarm: float = 1.0,
) -> float:
    """Compute the minimum detection cost function (minDCF).

    Args:
        scores: Verification scores; higher values mean more similar.
        labels: Binary labels with genuine ``1`` and impostor ``0``.
        target_prior: Prior probability of a target trial.
        c_miss: Cost of rejecting a genuine trial.
        c_false_alarm: Cost of accepting an impostor trial.

    Returns:
        The lowest normalized detection cost over all empirical thresholds.
    """
    if not 0.0 < target_prior < 1.0:
        raise ValueError("target_prior must be between 0 and 1")
    if c_miss < 0 or c_false_alarm < 0:
        raise ValueError("DCF costs must be non-negative")

    score_array, label_array = _validate_inputs(scores, labels)
    _, false_accept, false_reject = _operating_points(score_array, label_array)
    costs = (
        target_prior * c_miss * false_reject
        + (1.0 - target_prior) * c_false_alarm * false_accept
    )
    default_cost = min(target_prior * c_miss, (1.0 - target_prior) * c_false_alarm)
    if default_cost == 0:
        return float(np.min(costs))
    return float(np.min(costs) / default_cost)


def compute_accuracy(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    threshold: float = 0.0,
) -> float:
    """Compute binary verification accuracy at a supplied threshold."""
    score_array, label_array = _validate_inputs(scores, labels)
    predictions = (score_array >= threshold).astype(np.int64)
    return float(np.mean(predictions == label_array))


def compute_verification_metrics(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    target_prior: float = 0.01,
) -> dict[str, Any]:
    """Compute the standard RawNet3 verification metric set.

    Accuracy is reported at the empirical threshold closest to the EER
    operating point. The selected threshold is included for reproducibility.
    """
    score_array, label_array = _validate_inputs(scores, labels)
    eer, eer_threshold = _eer_details(score_array, label_array)
    return {
        "eer": eer,
        "eer_percent": eer * 100.0,
        "min_dcf": compute_min_dcf(
            score_array, label_array, target_prior=target_prior
        ),
        "accuracy": compute_accuracy(score_array, label_array, eer_threshold),
        "decision_threshold": eer_threshold,
        "num_trials": int(label_array.size),
        "num_genuine": int(np.sum(label_array == 1)),
        "num_impostor": int(np.sum(label_array == 0)),
        "target_prior": target_prior,
    }


def save_metrics(
    metrics: Mapping[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """Write metrics as indented JSON and return the output path.

    When no path is supplied, the result is written to the repository-local
    ``Thanh/results/rawnet3_metrics.json`` location regardless of the current
    working directory.
    """
    destination = DEFAULT_METRICS_PATH if output_path is None else Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as metrics_file:
        json.dump(dict(metrics), metrics_file, indent=2, sort_keys=True)
        metrics_file.write("\n")
    return destination
