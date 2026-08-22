"""Atomic, identity-guarded PyTorch training checkpoints for Kaggle."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - Kaggle dependency gate.
    raise ModuleNotFoundError(
        "Checkpointing requires Kaggle's CUDA-matched PyTorch build."
    ) from error

from speaker_recognition.training.lifecycle import (
    EpochSummary,
    TrainingCursor,
    TrainingLifecycleError,
    TrainingRunIdentity,
    validate_epoch_history,
)


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_PURPOSE = "speaker_training_exact_resume"


class CheckpointError(ValueError):
    """Raised when a checkpoint is corrupt or belongs to another run."""


@dataclass(frozen=True, slots=True)
class RestoredTrainingState:
    """Return the framework-neutral state recovered from model checkpointing."""

    cursor: TrainingCursor
    history: tuple[EpochSummary, ...]
    checkpoint_sha256: str


def save_training_checkpoint(
    path: str | Path,
    *,
    adapter: torch.nn.Module,
    objective: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    identity: TrainingRunIdentity,
    cursor: TrainingCursor,
    history: Sequence[EpochSummary],
    optimizer_group_names: Sequence[str],
    device: str,
) -> str:
    """Atomically persist exact resume state and an inspectable JSON sidecar.

    The binary is written to a sibling ``.part`` file, flushed to disk, then
    replaced atomically. A crash therefore leaves either the previous complete
    checkpoint or the new complete checkpoint, never a partially overwritten
    target file.
    """
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    group_names = _validate_group_names(optimizer_group_names)
    if len(group_names) != len(optimizer.param_groups):
        raise CheckpointError(
            "Optimizer group names must match optimizer parameter groups."
        )
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise CheckpointError("Accepted training checkpoints require CUDA.")

    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "purpose": CHECKPOINT_PURPOSE,
        "identity": identity.to_dict(),
        "cursor": cursor.to_dict(),
        "history": [summary.to_dict() for summary in history],
        "optimizer_group_names": list(group_names),
        "adapter_state": adapter.state_dict(),
        "objective_state": objective.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        # The sampler and cropper own private deterministic seeds. Global RNG
        # restoration is still required for dropout and layerdrop continuity.
        "python_rng_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state(device),
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": device,
        },
    }

    partial = destination.with_name(destination.name + ".part")
    try:
        with partial.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()

    checkpoint_sha256 = _sha256_file(destination)
    _write_json_atomic(
        destination.with_name(destination.name + ".json"),
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "purpose": "speaker_training_checkpoint_sidecar",
            "checkpoint_filename": destination.name,
            "checkpoint_sha256": checkpoint_sha256,
            "identity": identity.to_dict(),
            "cursor": cursor.to_dict(),
            "history": [summary.to_dict() for summary in history],
            "optimizer_group_names": list(group_names),
        },
    )
    return checkpoint_sha256


def restore_training_checkpoint(
    path: str | Path,
    *,
    adapter: torch.nn.Module,
    objective: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    expected_identity: TrainingRunIdentity,
    expected_optimizer_group_names: Sequence[str],
    device: str,
) -> RestoredTrainingState:
    """Strictly restore a checkpoint only when run identity matches exactly."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise CheckpointError(f"Checkpoint does not exist: {checkpoint_path}")
    expected_groups = _validate_group_names(expected_optimizer_group_names)
    checkpoint_sha256 = _sha256_file(checkpoint_path)

    payload: Any = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(payload, Mapping):
        raise CheckpointError("Checkpoint root must be a mapping.")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("Checkpoint schema version changed.")
    if payload.get("purpose") != CHECKPOINT_PURPOSE:
        raise CheckpointError("Checkpoint purpose changed.")
    if payload.get("identity") != expected_identity.to_dict():
        raise CheckpointError(
            "Checkpoint identity does not match model, dataset, config, "
            "manifest, and seed."
        )
    if tuple(payload.get("optimizer_group_names", ())) != expected_groups:
        raise CheckpointError("Checkpoint optimizer groups changed.")

    try:
        cursor_mapping = _required_mapping(payload, "cursor")
        history_values = payload.get("history")
        if not isinstance(history_values, list) or any(
            not isinstance(value, Mapping) for value in history_values
        ):
            raise CheckpointError("Checkpoint history must be a list of mappings.")
        cursor = TrainingCursor.from_mapping(cursor_mapping)
        history = validate_epoch_history(history_values)
    except TrainingLifecycleError as error:
        raise CheckpointError(f"Invalid checkpoint lifecycle state: {error}") from error

    _strict_load_module(adapter, payload.get("adapter_state"), "adapter")
    _strict_load_module(objective, payload.get("objective_state"), "objective")
    optimizer_state = payload.get("optimizer_state")
    scaler_state = payload.get("scaler_state")
    if not isinstance(optimizer_state, Mapping):
        raise CheckpointError("Checkpoint optimizer state must be a mapping.")
    if not isinstance(scaler_state, Mapping):
        raise CheckpointError("Checkpoint scaler state must be a mapping.")
    optimizer.load_state_dict(optimizer_state)
    scaler.load_state_dict(scaler_state)

    python_rng_state = payload.get("python_rng_state")
    cpu_rng_state = payload.get("torch_cpu_rng_state")
    cuda_rng_state = payload.get("torch_cuda_rng_state")
    if not isinstance(python_rng_state, tuple):
        raise CheckpointError("Checkpoint Python RNG state is invalid.")
    if not isinstance(cpu_rng_state, torch.Tensor):
        raise CheckpointError("Checkpoint CPU RNG state is invalid.")
    if not isinstance(cuda_rng_state, torch.Tensor):
        raise CheckpointError("Checkpoint CUDA RNG state is invalid.")
    random.setstate(python_rng_state)
    torch.set_rng_state(cpu_rng_state.cpu())
    torch.cuda.set_rng_state(cuda_rng_state.cpu(), device=device)

    return RestoredTrainingState(
        cursor=cursor,
        history=history,
        checkpoint_sha256=checkpoint_sha256,
    )


def _strict_load_module(
    module: torch.nn.Module,
    state: object,
    component: str,
) -> None:
    """Load a tensor state mapping and reject missing or unexpected keys."""
    if not isinstance(state, Mapping):
        raise CheckpointError(f"Checkpoint {component} state must be a mapping.")
    incompatible = module.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise CheckpointError(
            f"Strict {component} restore failed: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}."
        )


def _required_mapping(
    values: Mapping[str, Any],
    key: str,
) -> Mapping[str, object]:
    """Read one required checkpoint mapping."""
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise CheckpointError(f"Checkpoint {key} must be a mapping.")
    return value


def _validate_group_names(values: Sequence[str]) -> tuple[str, ...]:
    """Require unique non-empty optimizer group names in stable order."""
    names = tuple(values)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise CheckpointError("Optimizer group names must be non-empty strings.")
    if len(names) != len(set(names)):
        raise CheckpointError("Optimizer group names must be unique.")
    return names


def _sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash one checkpoint without materializing it in host memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write one finite, inspectable metadata sidecar transactionally."""
    partial = path.with_name(path.name + ".part")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()
