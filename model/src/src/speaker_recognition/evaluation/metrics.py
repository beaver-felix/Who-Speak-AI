"""Auditable speaker-verification metrics computed from trial scores."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


DEFAULT_FAR_TARGETS = (0.05, 0.01, 0.001, 0.0001)


class VerificationMetricError(ValueError):
    """Raised when verification scores cannot support valid metrics."""


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """Confusion counts and rates at one accept-if-score-at-least threshold."""

    threshold: float
    far: float
    frr: float
    tar: float
    accuracy: float
    true_accepts: int
    false_accepts: int
    true_rejects: int
    false_rejects: int


@dataclass(frozen=True, slots=True)
class RocCurve:
    """All distinct empirical verification operating points."""

    thresholds: NDArray[np.float64]
    fars: NDArray[np.float64]
    frrs: NDArray[np.float64]
    tars: NDArray[np.float64]
    accuracies: NDArray[np.float64]
    true_accepts: NDArray[np.int64]
    false_accepts: NDArray[np.int64]
    true_rejects: NDArray[np.int64]
    false_rejects: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class MinDcfResult:
    """Minimum normalized detection cost and its empirical threshold."""

    value: float
    unnormalized_cost: float
    threshold: float
    far: float
    frr: float
    p_target: float
    cost_miss: float
    cost_false_alarm: float


@dataclass(frozen=True, slots=True)
class VerificationMetrics:
    """Complete required metric set for one scored trial list."""

    eer: float
    eer_threshold: float
    min_dcf: MinDcfResult
    decision: OperatingPoint
    tar_at_far: dict[float, OperatingPoint]

    def to_flat_dict(self) -> dict[str, float | int]:
        """Return stable scalar names suitable for JSON and W&B logging."""
        values: dict[str, float | int] = {
            "eer": self.eer,
            "eer_threshold": self.eer_threshold,
            "min_dcf": self.min_dcf.value,
            "min_dcf_threshold": self.min_dcf.threshold,
            "decision_threshold": self.decision.threshold,
            "far": self.decision.far,
            "frr": self.decision.frr,
            "tar": self.decision.tar,
            "accuracy": self.decision.accuracy,
            "true_accepts": self.decision.true_accepts,
            "false_accepts": self.decision.false_accepts,
            "true_rejects": self.decision.true_rejects,
            "false_rejects": self.decision.false_rejects,
        }
        for target, point in sorted(self.tar_at_far.items()):
            suffix = _format_far_target(target)
            values[f"tar_at_far_{suffix}"] = point.tar
            values[f"achieved_far_at_target_{suffix}"] = point.far
            values[f"threshold_at_far_{suffix}"] = point.threshold
        return values


def build_roc_curve(labels: ArrayLike, scores: ArrayLike) -> RocCurve:
    """Build empirical ROC points while handling score ties atomically.

    Labels use ``1`` for genuine trials and ``0`` for impostor trials. A trial
    is accepted when its score is greater than or equal to the threshold.
    Tied scores are updated as one group so impossible partial-tie operating
    points are never reported.
    """
    trial_labels, trial_scores = _validated_trials(labels, scores)
    genuine_count = int(trial_labels.sum())
    impostor_count = trial_labels.size - genuine_count

    order = np.argsort(-trial_scores, kind="stable")
    sorted_scores = trial_scores[order]
    sorted_labels = trial_labels[order]

    thresholds = [float("inf")]
    true_accepts = [0]
    false_accepts = [0]
    accepted_genuine = 0
    accepted_impostor = 0
    index = 0

    while index < sorted_scores.size:
        score = sorted_scores[index]
        group_end = index + 1
        while (
            group_end < sorted_scores.size
            and sorted_scores[group_end] == score
        ):
            group_end += 1

        group_labels = sorted_labels[index:group_end]
        group_genuine = int(group_labels.sum())
        accepted_genuine += group_genuine
        accepted_impostor += group_labels.size - group_genuine
        thresholds.append(float(score))
        true_accepts.append(accepted_genuine)
        false_accepts.append(accepted_impostor)
        index = group_end

    true_accept_array = np.asarray(true_accepts, dtype=np.int64)
    false_accept_array = np.asarray(false_accepts, dtype=np.int64)
    false_reject_array = genuine_count - true_accept_array
    true_reject_array = impostor_count - false_accept_array
    far_array = false_accept_array.astype(np.float64) / impostor_count
    frr_array = false_reject_array.astype(np.float64) / genuine_count
    tar_array = true_accept_array.astype(np.float64) / genuine_count
    accuracy_array = (
        true_accept_array + true_reject_array
    ).astype(np.float64) / trial_labels.size

    return RocCurve(
        thresholds=np.asarray(thresholds, dtype=np.float64),
        fars=far_array,
        frrs=frr_array,
        tars=tar_array,
        accuracies=accuracy_array,
        true_accepts=true_accept_array,
        false_accepts=false_accept_array,
        true_rejects=true_reject_array,
        false_rejects=false_reject_array,
    )


def compute_eer(curve: RocCurve) -> tuple[float, float]:
    """Return linearly interpolated equal-error rate and threshold."""
    differences = curve.fars - curve.frrs
    exact_indices = np.flatnonzero(differences == 0.0)
    if exact_indices.size:
        index = int(exact_indices[0])
        return float(curve.fars[index]), float(curve.thresholds[index])

    crossing_index = int(np.flatnonzero(differences > 0.0)[0])
    previous_index = crossing_index - 1
    previous_difference = differences[previous_index]
    current_difference = differences[crossing_index]
    interpolation = -previous_difference / (
        current_difference - previous_difference
    )
    eer = curve.fars[previous_index] + interpolation * (
        curve.fars[crossing_index] - curve.fars[previous_index]
    )

    previous_threshold = curve.thresholds[previous_index]
    current_threshold = curve.thresholds[crossing_index]
    if np.isfinite(previous_threshold):
        threshold = previous_threshold + interpolation * (
            current_threshold - previous_threshold
        )
    else:
        # With one all-tied score, no finite threshold realizes the
        # interpolated point. Report the only empirical accept threshold.
        threshold = current_threshold

    return float(eer), float(threshold)


def compute_operating_point(
    labels: ArrayLike,
    scores: ArrayLike,
    *,
    threshold: float,
) -> OperatingPoint:
    """Compute FAR, FRR, TAR, and accuracy at an external threshold."""
    trial_labels, trial_scores = _validated_trials(labels, scores)
    if not isinstance(threshold, Real) or np.isnan(float(threshold)):
        raise VerificationMetricError("threshold must be a real non-NaN value.")

    accepted = trial_scores >= float(threshold)
    genuine = trial_labels == 1
    impostor = ~genuine
    true_accepts = int(np.count_nonzero(accepted & genuine))
    false_accepts = int(np.count_nonzero(accepted & impostor))
    false_rejects = int(np.count_nonzero(~accepted & genuine))
    true_rejects = int(np.count_nonzero(~accepted & impostor))
    return _make_operating_point(
        threshold=float(threshold),
        true_accepts=true_accepts,
        false_accepts=false_accepts,
        true_rejects=true_rejects,
        false_rejects=false_rejects,
    )


def compute_min_dcf(
    curve: RocCurve,
    *,
    p_target: float = 0.01,
    cost_miss: float = 1.0,
    cost_false_alarm: float = 1.0,
) -> MinDcfResult:
    """Compute NIST-style minimum normalized detection cost.

    The normalization denominator is the cost of the better trivial system:
    always reject or always accept.
    """
    if not 0.0 < p_target < 1.0:
        raise VerificationMetricError("p_target must lie strictly within (0, 1).")
    if cost_miss <= 0.0 or cost_false_alarm <= 0.0:
        raise VerificationMetricError("Detection costs must be positive.")

    costs = (
        cost_miss * p_target * curve.frrs
        + cost_false_alarm * (1.0 - p_target) * curve.fars
    )
    normalizer = min(
        cost_miss * p_target,
        cost_false_alarm * (1.0 - p_target),
    )
    normalized_costs = costs / normalizer
    index = int(np.argmin(normalized_costs))
    return MinDcfResult(
        value=float(normalized_costs[index]),
        unnormalized_cost=float(costs[index]),
        threshold=float(curve.thresholds[index]),
        far=float(curve.fars[index]),
        frr=float(curve.frrs[index]),
        p_target=p_target,
        cost_miss=cost_miss,
        cost_false_alarm=cost_false_alarm,
    )


def select_tar_at_far(curve: RocCurve, *, far_target: float) -> OperatingPoint:
    """Select maximum empirical TAR subject to FAR not exceeding a target."""
    if not 0.0 <= far_target <= 1.0:
        raise VerificationMetricError("far_target must lie within [0, 1].")

    eligible_indices = np.flatnonzero(curve.fars <= far_target)
    best_index = max(
        (int(index) for index in eligible_indices),
        key=lambda index: (
            curve.tars[index],
            -curve.fars[index],
            curve.thresholds[index],
        ),
    )
    return _point_from_curve(curve, best_index)


def compute_verification_metrics(
    labels: ArrayLike,
    scores: ArrayLike,
    *,
    decision_threshold: float,
    far_targets: tuple[float, ...] = DEFAULT_FAR_TARGETS,
    p_target: float = 0.01,
    cost_miss: float = 1.0,
    cost_false_alarm: float = 1.0,
) -> VerificationMetrics:
    """Compute every required metric using one explicit decision threshold."""
    curve = build_roc_curve(labels, scores)
    eer, eer_threshold = compute_eer(curve)
    min_dcf = compute_min_dcf(
        curve,
        p_target=p_target,
        cost_miss=cost_miss,
        cost_false_alarm=cost_false_alarm,
    )
    decision = compute_operating_point(
        labels,
        scores,
        threshold=decision_threshold,
    )
    tar_points = {
        target: select_tar_at_far(curve, far_target=target)
        for target in far_targets
    }
    return VerificationMetrics(
        eer=eer,
        eer_threshold=eer_threshold,
        min_dcf=min_dcf,
        decision=decision,
        tar_at_far=tar_points,
    )


def _validated_trials(
    labels: ArrayLike,
    scores: ArrayLike,
) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
    """Validate and normalize genuine/impostor trial arrays."""
    label_array = np.asarray(labels)
    score_array = np.asarray(scores, dtype=np.float64)
    if label_array.ndim != 1 or score_array.ndim != 1:
        raise VerificationMetricError("labels and scores must be one-dimensional.")
    if label_array.size == 0 or label_array.size != score_array.size:
        raise VerificationMetricError(
            "labels and scores must have equal non-zero length."
        )
    if not np.isfinite(score_array).all():
        raise VerificationMetricError("scores must all be finite.")
    if not np.isin(label_array, (0, 1)).all():
        raise VerificationMetricError("labels must contain only 0 and 1.")

    normalized_labels = label_array.astype(np.int8, copy=False)
    if normalized_labels.sum() == 0:
        raise VerificationMetricError("at least one genuine trial is required.")
    if normalized_labels.sum() == normalized_labels.size:
        raise VerificationMetricError("at least one impostor trial is required.")
    return normalized_labels, score_array


def _make_operating_point(
    *,
    threshold: float,
    true_accepts: int,
    false_accepts: int,
    true_rejects: int,
    false_rejects: int,
) -> OperatingPoint:
    """Build rates from internally consistent confusion counts."""
    genuine_count = true_accepts + false_rejects
    impostor_count = false_accepts + true_rejects
    total_count = genuine_count + impostor_count
    return OperatingPoint(
        threshold=threshold,
        far=false_accepts / impostor_count,
        frr=false_rejects / genuine_count,
        tar=true_accepts / genuine_count,
        accuracy=(true_accepts + true_rejects) / total_count,
        true_accepts=true_accepts,
        false_accepts=false_accepts,
        true_rejects=true_rejects,
        false_rejects=false_rejects,
    )


def _point_from_curve(curve: RocCurve, index: int) -> OperatingPoint:
    """Convert one empirical curve row to an operating point."""
    return OperatingPoint(
        threshold=float(curve.thresholds[index]),
        far=float(curve.fars[index]),
        frr=float(curve.frrs[index]),
        tar=float(curve.tars[index]),
        accuracy=float(curve.accuracies[index]),
        true_accepts=int(curve.true_accepts[index]),
        false_accepts=int(curve.false_accepts[index]),
        true_rejects=int(curve.true_rejects[index]),
        false_rejects=int(curve.false_rejects[index]),
    )


def _format_far_target(value: float) -> str:
    """Format a FAR target as a stable percentage-based metric suffix."""
    percentage = value * 100.0
    return f"{percentage:g}pct".replace(".", "p")
