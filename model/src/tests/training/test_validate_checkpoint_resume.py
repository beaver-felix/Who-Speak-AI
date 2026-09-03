"""Tests for strict CUDA checkpoint/resume evidence validation."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts/validate_checkpoint_resume.py"
CHECK_NAMES = {
    "checkpoint_hash_matches_restore",
    "sidecar_hash_matches_checkpoint",
    "restored_cursor_matches_saved_cursor",
    "restored_history_is_empty",
    "step_two_loss_exact",
    "step_two_accuracy_exact",
    "step_two_gradient_norm_exact",
    "step_two_loss_scale_exact",
    "final_cursor_exact",
    "adapter_state_exact",
    "objective_state_exact",
    "optimizer_state_exact",
    "scaler_state_exact",
    "rng_dependent_embedding_exact",
}


def _load_validator() -> ModuleType:
    """Load the standalone validator without package-side imports."""
    specification = importlib.util.spec_from_file_location(
        "validate_checkpoint_resume",
        SCRIPT_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _cursor(step: int) -> dict[str, object]:
    """Return one valid fixture cursor after ``step`` optimizer updates."""
    return {
        "epoch_index": 0,
        "next_batch_index": step,
        "global_step": step,
        "epoch_loss_sum": 8.0 * step,
        "epoch_accuracy_sum": 0.0,
        "epoch_example_count": 4 * step,
        "best_validation_eer": None,
        "best_validation_min_dcf": None,
        "best_epoch_index": None,
        "stale_validation_count": 0,
    }


def _payload() -> dict[str, object]:
    """Return minimal complete equivalence evidence."""
    step = {
        "loss": 2.0,
        "accuracy": 0.0,
        "gradient_norm_before_clipping": 3.0,
        "loss_scale": 1024.0,
    }
    hashes = {
        "adapter": "1" * 64,
        "objective": "2" * 64,
        "optimizer": "3" * 64,
        "scaler": "4" * 64,
        "embeddings": "5" * 64,
    }
    return {
        "schema_version": 1,
        "purpose": "cuda_checkpoint_interruption_resume_equivalence",
        "fixture": {
            "seed": 42,
            "batch_size": 4,
            "input_dim": 16,
            "embedding_dim": 8,
            "class_count": 6,
            "dropout_probability": 0.25,
            "optimizer": "adamw",
            "optimizer_group_names": ["encoder", "aam_softmax_head"],
            "mixed_precision": "fp16",
            "gradient_clip_norm": 5.0,
            "initial_loss_scale": 1024.0,
        },
        "identity": {
            "model_name": "checkpoint_fixture",
            "dataset_name": "synthetic",
            "config_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "seed": 42,
        },
        "runtime": {
            "torch_version": "2.10.0+cu128",
            "cuda_version": "12.8",
            "device": "cuda:0",
            "device_name": "Tesla T4",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        },
        "checkpoint": {
            "schema_version": 1,
            "filename": "fixture_resume.pt",
            "sha256": "c" * 64,
            "sidecar_filename": "fixture_resume.pt.json",
        },
        "first_step": step,
        "control_second_step": step,
        "resumed_second_step": step,
        "control_state_sha256": dict(hashes),
        "resumed_state_sha256": dict(hashes),
        "saved_cursor": _cursor(1),
        "control_final_cursor": _cursor(2),
        "resumed_final_cursor": _cursor(2),
        "checks": {name: True for name in CHECK_NAMES},
    }


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    """Write one finite validator fixture."""
    path = tmp_path / "checkpoint_resume.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_accept_exact_checkpoint_resume_evidence(tmp_path: Path) -> None:
    """Complete deterministic equivalence evidence should pass."""
    summary = _load_validator().validate_artifact(
        _write(tmp_path, _payload())
    )

    assert summary == {
        "checkpoint_sha256": "c" * 64,
        "checks": 14,
        "step_two_loss": 2.0,
    }


def test_accept_committed_kaggle_checkpoint_evidence() -> None:
    """The accepted real CUDA artifact must remain validator-clean."""
    artifact = (
        PROJECT_ROOT
        / "results/model_audit/checkpoint_resume_gate.json"
    )

    assert _load_validator().validate_artifact(artifact) == {
        "checkpoint_sha256": (
            "b3aaf65f6f3b7616a73396a247355e4916bd03edeb2637ee431cd6df95be456c"
        ),
        "checks": 14,
        "step_two_loss": pytest.approx(17.66690444946289),
    }


def test_reject_changed_resumed_state_hash(tmp_path: Path) -> None:
    """One mismatched optimizer state must fail the entire gate."""
    payload = deepcopy(_payload())
    payload["resumed_state_sha256"]["optimizer"] = "9" * 64  # type: ignore[index]

    with pytest.raises(ValueError, match="state hashes differ"):
        _load_validator().validate_artifact(_write(tmp_path, payload))


def test_reject_nonfinite_resumed_metric(tmp_path: Path) -> None:
    """NaN must fail even when every boolean check claims success."""
    payload = deepcopy(_payload())
    payload["resumed_second_step"]["loss"] = float("nan")  # type: ignore[index]

    with pytest.raises(ValueError, match="must be finite"):
        _load_validator().validate_artifact(_write(tmp_path, payload))


def test_reject_false_equivalence_check(tmp_path: Path) -> None:
    """A skipped or failed exactness assertion must fail closed."""
    payload = deepcopy(_payload())
    payload["checks"]["rng_dependent_embedding_exact"] = False  # type: ignore[index]

    with pytest.raises(ValueError, match="checks failed"):
        _load_validator().validate_artifact(_write(tmp_path, payload))
