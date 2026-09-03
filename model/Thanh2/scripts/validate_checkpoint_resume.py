"""Validate CUDA checkpoint/resume evidence without importing PyTorch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping


EXPECTED_CHECKS = {
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
STATE_COMPONENTS = {
    "adapter",
    "objective",
    "optimizer",
    "scaler",
    "embeddings",
}


def parse_arguments() -> argparse.Namespace:
    """Parse one downloaded checkpoint-equivalence JSON artifact."""
    parser = argparse.ArgumentParser(
        description="Validate CUDA checkpoint/resume evidence."
    )
    parser.add_argument("artifact", type=Path)
    return parser.parse_args()


def validate_artifact(path: Path) -> dict[str, object]:
    """Fail closed on incomplete, non-finite, or non-equivalent evidence."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("Checkpoint evidence schema_version must equal 1.")
    if payload.get("purpose") != (
        "cuda_checkpoint_interruption_resume_equivalence"
    ):
        raise ValueError("Checkpoint evidence purpose changed.")

    fixture = _mapping(payload, "fixture")
    expected_fixture = {
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
    }
    if dict(fixture) != expected_fixture:
        raise ValueError("Checkpoint fixture configuration changed.")

    identity = _mapping(payload, "identity")
    expected_identity = {
        "model_name": "checkpoint_fixture",
        "dataset_name": "synthetic",
        "config_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "seed": 42,
    }
    if dict(identity) != expected_identity:
        raise ValueError("Checkpoint run identity changed.")

    runtime = _mapping(payload, "runtime")
    if runtime.get("device_name") != "Tesla T4":
        raise ValueError("Checkpoint evidence must come from a Tesla T4.")
    if runtime.get("torch_version") != "2.10.0+cu128":
        raise ValueError("Checkpoint evidence used an unexpected PyTorch.")
    if runtime.get("cuda_version") != "12.8":
        raise ValueError("Checkpoint evidence used an unexpected CUDA.")
    if runtime.get("device") != "cuda:0":
        raise ValueError("Checkpoint evidence must use cuda:0.")
    if runtime.get("deterministic_algorithms") is not True:
        raise ValueError("Deterministic algorithms were not enabled.")
    if runtime.get("cublas_workspace_config") != ":4096:8":
        raise ValueError("Deterministic cuBLAS workspace policy changed.")

    checkpoint = _mapping(payload, "checkpoint")
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Training checkpoint schema changed.")
    checkpoint_sha256 = checkpoint.get("sha256")
    if not _is_sha256(checkpoint_sha256):
        raise ValueError("Training checkpoint SHA-256 is invalid.")

    first_step = _mapping(payload, "first_step")
    control_step = _mapping(payload, "control_second_step")
    resumed_step = _mapping(payload, "resumed_second_step")
    _validate_step(first_step, "first_step")
    _validate_step(control_step, "control_second_step")
    _validate_step(resumed_step, "resumed_second_step")
    if dict(control_step) != dict(resumed_step):
        raise ValueError("Resumed step metrics differ from the control.")

    control_hashes = _mapping(payload, "control_state_sha256")
    resumed_hashes = _mapping(payload, "resumed_state_sha256")
    if set(control_hashes) != STATE_COMPONENTS:
        raise ValueError("Control state hash components changed.")
    if set(resumed_hashes) != STATE_COMPONENTS:
        raise ValueError("Resumed state hash components changed.")
    if any(not _is_sha256(value) for value in control_hashes.values()):
        raise ValueError("Control state contains an invalid SHA-256.")
    if dict(control_hashes) != dict(resumed_hashes):
        raise ValueError("Resumed state hashes differ from the control.")

    saved_cursor = _mapping(payload, "saved_cursor")
    control_cursor = _mapping(payload, "control_final_cursor")
    resumed_cursor = _mapping(payload, "resumed_final_cursor")
    _validate_cursor(saved_cursor, global_step=1, next_batch_index=1)
    _validate_cursor(control_cursor, global_step=2, next_batch_index=2)
    if dict(control_cursor) != dict(resumed_cursor):
        raise ValueError("Resumed final cursor differs from the control.")

    checks = _mapping(payload, "checks")
    if set(checks) != EXPECTED_CHECKS or not all(
        checks.get(name) is True for name in EXPECTED_CHECKS
    ):
        raise ValueError("One or more checkpoint equivalence checks failed.")
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "checks": len(EXPECTED_CHECKS),
        "step_two_loss": float(control_step["loss"]),
    }


def _validate_step(values: Mapping[str, object], name: str) -> None:
    """Require exactly the finite scalar metrics emitted by one step."""
    expected_fields = {
        "loss",
        "accuracy",
        "gradient_norm_before_clipping",
        "loss_scale",
    }
    if set(values) != expected_fields:
        raise ValueError(f"{name} metric fields changed.")
    for field, value in values.items():
        if not _finite_number(value):
            raise ValueError(f"{name}.{field} must be finite.")
    if not 0.0 <= float(values["accuracy"]) <= 1.0:
        raise ValueError(f"{name}.accuracy must be in [0, 1].")
    if float(values["loss"]) < 0.0:
        raise ValueError(f"{name}.loss must be non-negative.")
    if float(values["gradient_norm_before_clipping"]) < 0.0:
        raise ValueError(f"{name}.gradient norm must be non-negative.")
    if float(values["loss_scale"]) != 1024.0:
        raise ValueError(f"{name}.loss scale changed.")


def _validate_cursor(
    values: Mapping[str, object],
    *,
    global_step: int,
    next_batch_index: int,
) -> None:
    """Require the exact bounded cursor position while allowing metric sums."""
    required_fields = {
        "epoch_index",
        "next_batch_index",
        "global_step",
        "epoch_loss_sum",
        "epoch_accuracy_sum",
        "epoch_example_count",
        "best_validation_eer",
        "best_validation_min_dcf",
        "best_epoch_index",
        "stale_validation_count",
    }
    if set(values) != required_fields:
        raise ValueError("Checkpoint cursor fields changed.")
    if values.get("epoch_index") != 0:
        raise ValueError("Fixture must remain inside epoch zero.")
    if values.get("global_step") != global_step:
        raise ValueError("Fixture global step changed.")
    if values.get("next_batch_index") != next_batch_index:
        raise ValueError("Fixture next batch changed.")
    if values.get("epoch_example_count") != global_step * 4:
        raise ValueError("Fixture example count changed.")
    if any(
        values.get(field) is not None
        for field in (
            "best_validation_eer",
            "best_validation_min_dcf",
            "best_epoch_index",
        )
    ):
        raise ValueError("Fixture must not contain Validation selection state.")
    if values.get("stale_validation_count") != 0:
        raise ValueError("Fixture stale-validation count changed.")
    for field in ("epoch_loss_sum", "epoch_accuracy_sum"):
        if not _finite_number(values.get(field)):
            raise ValueError(f"Fixture {field} must be finite.")


def _mapping(
    values: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    """Read one required mapping from the evidence artifact."""
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Checkpoint evidence {key} must be a mapping.")
    return value


def _finite_number(value: object) -> bool:
    """Return whether a value is numeric, non-boolean, and finite."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_sha256(value: object) -> bool:
    """Return whether a value is a lowercase hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and value.lower() == value
        and all(character in "0123456789abcdef" for character in value)
    )


def main() -> None:
    """Validate one artifact and print its compact acceptance summary."""
    summary = validate_artifact(parse_arguments().artifact.resolve())
    print("CUDA CHECKPOINT INTERRUPTION/RESUME EVIDENCE VALIDATED")
    print(f"checkpoint SHA-256: {summary['checkpoint_sha256']}")
    print(f"checks: {summary['checks']}")
    print(f"step-two loss: {summary['step_two_loss']:.8f}")


if __name__ == "__main__":
    main()
