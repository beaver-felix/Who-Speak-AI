"""Shared speaker-verification metrics for the WavLM pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

THANH_ROOT = Path(__file__).resolve().parent.parent
if str(THANH_ROOT) not in sys.path:
    sys.path.insert(0, str(THANH_ROOT))

from RawNet3.metrics import (  # noqa: E402
    REQUIRED_CSV_FIELDS,
    compute_accuracy,
    compute_eer,
    compute_far_frr,
    compute_min_dcf,
    compute_tar_at_far,
    compute_verification_metrics,
    save_evaluation_csv,
    save_metrics,
)

__all__ = [
    "REQUIRED_CSV_FIELDS",
    "compute_accuracy",
    "compute_eer",
    "compute_far_frr",
    "compute_min_dcf",
    "compute_tar_at_far",
    "compute_verification_metrics",
    "save_evaluation_csv",
    "save_metrics",
]
