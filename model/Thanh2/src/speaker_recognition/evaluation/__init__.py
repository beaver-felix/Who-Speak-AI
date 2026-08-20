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

__all__ = [
    "DEFAULT_FAR_TARGETS",
    "MinDcfResult",
    "OperatingPoint",
    "VerificationMetrics",
    "build_roc_curve",
    "compute_eer",
    "compute_min_dcf",
    "compute_operating_point",
    "compute_verification_metrics",
    "select_tar_at_far",
]
