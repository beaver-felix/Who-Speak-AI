"""Speaker-verification trial construction and metric utilities."""

from speaker_recognition.evaluation.metrics import (
    DEFAULT_FAR_TARGETS,
    MinDcfResult,
    OperatingPoint,
    VerificationMetrics,
    build_roc_curve,
    compute_eer,
    compute_min_dcf,
    compute_operating_point,
    compute_verification_metrics,
    select_tar_at_far,
)
from speaker_recognition.evaluation.embeddings import (
    EmbeddingEvaluationError,
    EmbeddingTable,
    ScoredTrialSet,
    aggregate_crop_embeddings,
    score_verification_trials,
)
from speaker_recognition.evaluation.protocol import (
    CachedEvaluationResult,
    evaluate_embedding_table,
)
from speaker_recognition.evaluation.trials import (
    TrialConstructionError,
    VerificationTrial,
    build_verification_trials,
    trial_list_sha256,
)

__all__ = [
    "DEFAULT_FAR_TARGETS",
    "CachedEvaluationResult",
    "EmbeddingEvaluationError",
    "EmbeddingTable",
    "MinDcfResult",
    "OperatingPoint",
    "ScoredTrialSet",
    "TrialConstructionError",
    "VerificationTrial",
    "VerificationMetrics",
    "aggregate_crop_embeddings",
    "build_roc_curve",
    "build_verification_trials",
    "compute_eer",
    "compute_min_dcf",
    "compute_operating_point",
    "compute_verification_metrics",
    "evaluate_embedding_table",
    "select_tar_at_far",
    "score_verification_trials",
    "trial_list_sha256",
]
