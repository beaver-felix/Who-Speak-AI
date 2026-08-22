"""Framework-neutral evaluation of immutable trials from cached embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Sequence

from speaker_recognition.evaluation.embeddings import (
    EmbeddingEvaluationError,
    EmbeddingTable,
    score_verification_trials,
)
from speaker_recognition.evaluation.metrics import (
    VerificationMetrics,
    build_roc_curve,
    compute_eer,
    compute_verification_metrics,
)
from speaker_recognition.evaluation.trials import VerificationTrial


@dataclass(frozen=True, slots=True)
class CachedEvaluationResult:
    """Bind metric results to exact trial and embedding fingerprints."""

    metrics: VerificationMetrics
    trial_list_sha256: str
    embedding_table_sha256: str
    trial_count: int
    genuine_trial_count: int
    impostor_trial_count: int
    utterance_count: int
    embedding_dim: int
    decision_threshold_source: str

    def to_artifact(self) -> dict[str, object]:
        """Return a finite JSON-compatible scientific evidence payload."""
        values: dict[str, object] = {
            "schema_version": 1,
            "protocol": {
                "trial_list_sha256": self.trial_list_sha256,
                "trial_count": self.trial_count,
                "genuine_trial_count": self.genuine_trial_count,
                "impostor_trial_count": self.impostor_trial_count,
            },
            "embeddings": {
                "embedding_table_sha256": self.embedding_table_sha256,
                "utterance_count": self.utterance_count,
                "embedding_dim": self.embedding_dim,
                "aggregation": "arithmetic_mean_then_l2_normalize",
                "score": "cosine_similarity",
            },
            "threshold_policy": {
                "decision_threshold_source": self.decision_threshold_source,
                "accept_rule": "cosine_score_greater_than_or_equal_threshold",
                "security_threshold_selected": False,
            },
            "metrics": self.metrics.to_flat_dict(),
            "min_dcf_configuration": {
                "p_target": self.metrics.min_dcf.p_target,
                "cost_miss": self.metrics.min_dcf.cost_miss,
                "cost_false_alarm": self.metrics.min_dcf.cost_false_alarm,
            },
        }
        _require_finite_json_numbers(values)
        return values


def evaluate_embedding_table(
    table: EmbeddingTable,
    trials: Sequence[VerificationTrial],
    *,
    expected_trial_sha256: str,
    decision_threshold: float | None = None,
) -> CachedEvaluationResult:
    """Score one cache and compute every required verification metric.

    When no external threshold is supplied, Validation's interpolated EER
    threshold is used only to report diagnostic FAR/FRR/TAR/accuracy. Final
    Test evaluation must instead supply the frozen threshold selected from
    Validation under the separately declared security policy.
    """
    scored = score_verification_trials(table, trials)
    if scored.trial_list_sha256 != expected_trial_sha256:
        raise EmbeddingEvaluationError(
            "Trial-list fingerprint differs from the declared protocol."
        )

    if decision_threshold is None:
        curve = build_roc_curve(scored.labels, scored.scores)
        _, selected_threshold = compute_eer(curve)
        threshold_source = "validation_interpolated_eer_diagnostic"
    else:
        if (
            isinstance(decision_threshold, bool)
            or not isinstance(decision_threshold, Real)
            or not math.isfinite(float(decision_threshold))
        ):
            raise EmbeddingEvaluationError(
                "External decision threshold must be finite and real."
            )
        selected_threshold = float(decision_threshold)
        threshold_source = "frozen_validation_security_threshold"

    metrics = compute_verification_metrics(
        scored.labels,
        scored.scores,
        decision_threshold=selected_threshold,
    )
    genuine_count = int(scored.labels.sum())
    return CachedEvaluationResult(
        metrics=metrics,
        trial_list_sha256=scored.trial_list_sha256,
        embedding_table_sha256=table.sha256,
        trial_count=int(scored.labels.size),
        genuine_trial_count=genuine_count,
        impostor_trial_count=int(scored.labels.size) - genuine_count,
        utterance_count=len(table.utterance_ids),
        embedding_dim=table.embedding_dim,
        decision_threshold_source=threshold_source,
    )


def _require_finite_json_numbers(value: object) -> None:
    """Recursively reject NaN and infinity before evidence serialization."""
    if isinstance(value, dict):
        for nested in value.values():
            _require_finite_json_numbers(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _require_finite_json_numbers(nested)
    elif isinstance(value, Real) and not isinstance(value, (bool, int)):
        if not math.isfinite(float(value)):
            raise EmbeddingEvaluationError(
                "Evaluation evidence contains a non-finite number."
            )
