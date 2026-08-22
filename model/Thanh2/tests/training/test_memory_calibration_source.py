"""Dependency-free checks for the Kaggle memory calibration gate."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "calibrate_training_memory.py"


def test_calibration_script_parses_without_model_dependencies() -> None:
    """The committed Kaggle entry point must be valid Python."""
    ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"), filename=str(SCRIPT_PATH))


def test_calibration_measures_complete_optimizer_step_memory() -> None:
    """Evidence must include loss, backward, clipping, optimizer, and CUDA peaks."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    for required in (
        "AamSoftmaxHead",
        "build_optimizer",
        ".backward()",
        "clip_grad_norm_",
        "scaler.step(optimizer)",
        "max_memory_allocated",
        "max_memory_reserved",
        "torch.OutOfMemoryError",
    ):
        assert required in source


def test_calibration_does_not_claim_to_select_batch_size() -> None:
    """A single repeated sample is a capacity gate, not performance evidence."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "memory_calibration_not_model_selection" in source
    assert "multi-batch mini-run" in source
