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
from speaker_recognition.evaluation.trials import (
    TrialConstructionError,
    VerificationTrial,
    build_verification_trials,
    trial_list_sha256,
)

__all__ = [
    "DEFAULT_FAR_TARGETS",
    "MinDcfResult",
    "OperatingPoint",
    "TrialConstructionError",
    "VerificationTrial",
    "VerificationMetrics",
    "build_roc_curve",
    "build_verification_trials",
    "compute_eer",
    "compute_min_dcf",
    "compute_operating_point",
    "compute_verification_metrics",
    "select_tar_at_far",
    "trial_list_sha256",
]
