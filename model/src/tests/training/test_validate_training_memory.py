"""Tests for strict dependency-free memory-evidence validation."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validate_training_memory.py"
EVIDENCE_ROOT = PROJECT_ROOT / "results" / "model_audit" / "training_memory"


def _load_validator() -> ModuleType:
    """Load the standalone validation script as a test module."""
    specification = importlib.util.spec_from_file_location(
        "validate_training_memory",
        SCRIPT_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _payload() -> dict[str, object]:
    """Return one minimal valid ECAPA schema-2 artifact."""
    return {
        "schema_version": 2,
        "purpose": "memory_calibration_not_model_selection",
        "model": {
            "name": "ecapa_tdnn",
            "embedding_dim": 192,
            "parameter_count": 20_767_552,
        },
        "objective": {
            "num_classes": 10_291,
            "margin": 0.2,
            "scale": 30.0,
        },
        "measurements": [
            {
                "batch_size": 16,
                "status": "passed",
                "loss": 12.5,
                "gradient_norm_before_clipping": 45.0,
                "loss_finite": True,
                "gradient_norm_finite": True,
                "optimizer_step_applied": True,
                "peak_allocated_bytes": 1_000_000,
                "peak_reserved_bytes": 2_000_000,
            }
        ],
        "largest_passing_batch_size": 16,
    }


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    """Write one JSON artifact fixture."""
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_accept_complete_finite_optimizer_step(tmp_path: Path) -> None:
    """A complete finite measurement should produce the 80% candidate."""
    summary = _load_validator().validate_artifact(
        _write(tmp_path, _payload())
    )

    assert summary == {
        "model": "ecapa_tdnn",
        "largest_passing_batch_size": 16,
        "conservative_candidate_batch_size": 12,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gradient_norm_before_clipping", float("inf"), "must be finite"),
        ("gradient_norm_finite", False, "gradient finite check failed"),
        ("optimizer_step_applied", False, "optimizer step was not applied"),
    ],
)
def test_reject_nonfinite_or_skipped_step(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """The issue found in schema-1 evidence must now fail closed."""
    payload = deepcopy(_payload())
    payload["measurements"][0][field] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=message):
        _load_validator().validate_artifact(_write(tmp_path, payload))


def test_reject_old_capacity_only_schema(tmp_path: Path) -> None:
    """Schema 1 cannot be mislabeled as validated optimizer-step evidence."""
    payload = _payload()
    payload["schema_version"] = 1

    with pytest.raises(ValueError, match="schema_version"):
        _load_validator().validate_artifact(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("filename", "model", "largest", "candidate"),
    [
        ("ecapa_tdnn_vimd_classes_t4.json", "ecapa_tdnn", 32, 25),
        ("rawnet3_vimd_classes_t4.json", "rawnet3", 32, 25),
        ("wavlm_mhfa_vimd_classes_t4.json", "wavlm_mhfa", 8, 6),
    ],
)
def test_committed_schema2_evidence_passes_strict_validation(
    filename: str,
    model: str,
    largest: int,
    candidate: int,
) -> None:
    """Accepted real T4 artifacts must remain tied to the strict validator."""
    summary = _load_validator().validate_artifact(EVIDENCE_ROOT / filename)

    assert summary == {
        "model": model,
        "largest_passing_batch_size": largest,
        "conservative_candidate_batch_size": candidate,
    }
