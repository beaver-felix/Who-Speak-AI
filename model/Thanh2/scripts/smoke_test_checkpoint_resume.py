"""Prove exact CUDA checkpoint/resume equivalence on a deterministic fixture."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

from speaker_recognition.training.lifecycle import (
    TrainingCursor,
    TrainingRunIdentity,
    record_training_batch,
)


FIXTURE_SEED = 42
FIXTURE_CONFIG_SHA256 = "a" * 64
FIXTURE_MANIFEST_SHA256 = "b" * 64
FIXTURE_BATCH_SIZE = 4
FIXTURE_INPUT_DIM = 16
FIXTURE_EMBEDDING_DIM = 8
FIXTURE_CLASS_COUNT = 6
GROUP_NAMES = ("encoder", "aam_softmax_head")


def parse_arguments() -> argparse.Namespace:
    """Parse a writable gate directory, CUDA device, and JSON destination."""
    parser = argparse.ArgumentParser(
        description="Smoke-test exact CUDA checkpoint interruption/resume."
    )
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Compare uninterrupted step two with restored step two exactly."""
    arguments = parse_arguments()
    # cuBLAS must see this before its first workspace is created.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch

    from speaker_recognition.training.checkpointing import (
        restore_training_checkpoint,
        save_training_checkpoint,
    )
    from speaker_recognition.training.objectives import AamSoftmaxHead

    if not torch.cuda.is_available() or not arguments.device.startswith("cuda"):
        raise RuntimeError("Checkpoint equivalence requires a CUDA device.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(FIXTURE_SEED)
    torch.manual_seed(FIXTURE_SEED)
    torch.cuda.manual_seed_all(FIXTURE_SEED)

    work_dir = arguments.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = work_dir / "fixture_resume.pt"
    identity = TrainingRunIdentity(
        model_name="checkpoint_fixture",
        dataset_name="synthetic",
        config_sha256=FIXTURE_CONFIG_SHA256,
        manifest_sha256=FIXTURE_MANIFEST_SHA256,
        seed=FIXTURE_SEED,
    )
    first_batch, second_batch = _fixture_batches(torch)
    adapter_class = _build_adapter_class(torch)

    adapter, objective, optimizer, scaler = _build_components(
        torch,
        AamSoftmaxHead,
        adapter_class,
        arguments.device,
    )
    cursor = TrainingCursor()
    first_metrics = _training_step(
        torch,
        adapter,
        objective,
        optimizer,
        scaler,
        first_batch,
        arguments.device,
    )
    cursor = record_training_batch(
        cursor,
        batch_size=FIXTURE_BATCH_SIZE,
        loss=first_metrics["loss"],
        accuracy=first_metrics["accuracy"],
    )
    saved_checkpoint_sha256 = save_training_checkpoint(
        checkpoint_path,
        adapter=adapter,
        objective=objective,
        optimizer=optimizer,
        scaler=scaler,
        identity=identity,
        cursor=cursor,
        history=(),
        optimizer_group_names=GROUP_NAMES,
        device=arguments.device,
    )
    saved_cursor = cursor

    control_metrics = _training_step(
        torch,
        adapter,
        objective,
        optimizer,
        scaler,
        second_batch,
        arguments.device,
    )
    control_cursor = record_training_batch(
        cursor,
        batch_size=FIXTURE_BATCH_SIZE,
        loss=control_metrics["loss"],
        accuracy=control_metrics["accuracy"],
    )
    control_hashes = _component_hashes(
        torch,
        adapter,
        objective,
        optimizer,
        scaler,
        control_metrics["embeddings"],
    )

    del adapter, objective, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()

    resumed_adapter, resumed_objective, resumed_optimizer, resumed_scaler = (
        _build_components(
            torch,
            AamSoftmaxHead,
            adapter_class,
            arguments.device,
        )
    )
    restored = restore_training_checkpoint(
        checkpoint_path,
        adapter=resumed_adapter,
        objective=resumed_objective,
        optimizer=resumed_optimizer,
        scaler=resumed_scaler,
        expected_identity=identity,
        expected_optimizer_group_names=GROUP_NAMES,
        device=arguments.device,
    )
    if restored.cursor != saved_cursor or restored.history:
        raise RuntimeError("Restored lifecycle state differs from the checkpoint.")

    resumed_metrics = _training_step(
        torch,
        resumed_adapter,
        resumed_objective,
        resumed_optimizer,
        resumed_scaler,
        second_batch,
        arguments.device,
    )
    resumed_cursor = record_training_batch(
        restored.cursor,
        batch_size=FIXTURE_BATCH_SIZE,
        loss=resumed_metrics["loss"],
        accuracy=resumed_metrics["accuracy"],
    )
    resumed_hashes = _component_hashes(
        torch,
        resumed_adapter,
        resumed_objective,
        resumed_optimizer,
        resumed_scaler,
        resumed_metrics["embeddings"],
    )

    sidecar_path = checkpoint_path.with_name(checkpoint_path.name + ".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    checks = {
        "checkpoint_hash_matches_restore": (
            restored.checkpoint_sha256 == saved_checkpoint_sha256
        ),
        "sidecar_hash_matches_checkpoint": (
            sidecar.get("checkpoint_sha256") == saved_checkpoint_sha256
        ),
        "restored_cursor_matches_saved_cursor": restored.cursor == saved_cursor,
        "restored_history_is_empty": not restored.history,
        "step_two_loss_exact": (
            resumed_metrics["loss"] == control_metrics["loss"]
        ),
        "step_two_accuracy_exact": (
            resumed_metrics["accuracy"] == control_metrics["accuracy"]
        ),
        "step_two_gradient_norm_exact": (
            resumed_metrics["gradient_norm_before_clipping"]
            == control_metrics["gradient_norm_before_clipping"]
        ),
        "step_two_loss_scale_exact": (
            resumed_metrics["loss_scale"] == control_metrics["loss_scale"]
        ),
        "final_cursor_exact": resumed_cursor == control_cursor,
        "adapter_state_exact": (
            resumed_hashes["adapter"] == control_hashes["adapter"]
        ),
        "objective_state_exact": (
            resumed_hashes["objective"] == control_hashes["objective"]
        ),
        "optimizer_state_exact": (
            resumed_hashes["optimizer"] == control_hashes["optimizer"]
        ),
        "scaler_state_exact": (
            resumed_hashes["scaler"] == control_hashes["scaler"]
        ),
        "rng_dependent_embedding_exact": (
            resumed_hashes["embeddings"] == control_hashes["embeddings"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Checkpoint/resume checks failed: {checks}")

    payload = {
        "schema_version": 1,
        "purpose": "cuda_checkpoint_interruption_resume_equivalence",
        "fixture": {
            "seed": FIXTURE_SEED,
            "batch_size": FIXTURE_BATCH_SIZE,
            "input_dim": FIXTURE_INPUT_DIM,
            "embedding_dim": FIXTURE_EMBEDDING_DIM,
            "class_count": FIXTURE_CLASS_COUNT,
            "dropout_probability": 0.25,
            "optimizer": "adamw",
            "optimizer_group_names": list(GROUP_NAMES),
            "mixed_precision": "fp16",
            "gradient_clip_norm": 5.0,
            "initial_loss_scale": 1024.0,
        },
        "identity": identity.to_dict(),
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": arguments.device,
            "device_name": torch.cuda.get_device_name(arguments.device),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
        },
        "checkpoint": {
            "schema_version": 1,
            "filename": checkpoint_path.name,
            "sha256": saved_checkpoint_sha256,
            "sidecar_filename": sidecar_path.name,
        },
        "first_step": _serializable_metrics(first_metrics),
        "control_second_step": _serializable_metrics(control_metrics),
        "resumed_second_step": _serializable_metrics(resumed_metrics),
        "control_state_sha256": control_hashes,
        "resumed_state_sha256": resumed_hashes,
        "saved_cursor": saved_cursor.to_dict(),
        "control_final_cursor": control_cursor.to_dict(),
        "resumed_final_cursor": resumed_cursor.to_dict(),
        "checks": checks,
    }
    _write_json(arguments.output, payload)
    print("CUDA CHECKPOINT INTERRUPTION/RESUME GATE PASSED")
    print(f"checkpoint SHA-256: {saved_checkpoint_sha256}")
    print(f"step-two loss: {control_metrics['loss']:.8f}")
    print(f"checks: {len(checks)}")
    print(f"output: {arguments.output.expanduser().resolve()}")


def _build_adapter_class(torch: Any) -> type:
    """Create a small dropout-bearing embedding adapter class."""

    class FixtureAdapter(torch.nn.Module):
        """Exercise trainable tensors and CUDA RNG without external weights."""

        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(FIXTURE_INPUT_DIM, 32),
                torch.nn.GELU(),
                torch.nn.Dropout(p=0.25),
                torch.nn.Linear(32, FIXTURE_EMBEDDING_DIM),
            )

        def forward(self, waveforms: Any) -> Any:
            embeddings = self.network(waveforms)
            return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return FixtureAdapter


def _build_components(
    torch: Any,
    objective_class: type,
    adapter_class: type,
    device: str,
) -> tuple[Any, Any, Any, Any]:
    """Construct newly initialized model, head, optimizer, and GradScaler."""
    adapter = adapter_class().to(device)
    objective = objective_class(
        embedding_dim=FIXTURE_EMBEDDING_DIM,
        num_classes=FIXTURE_CLASS_COUNT,
        margin=0.2,
        scale=30.0,
        easy_margin=False,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(adapter.parameters()), "lr": 1e-3},
            {"params": list(objective.parameters()), "lr": 2e-3},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
        init_scale=1024.0,
        growth_interval=100_000,
    )
    adapter.train()
    objective.train()
    return adapter, objective, optimizer, scaler


def _fixture_batches(torch: Any) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    """Return two fixed batches without consuming any global RNG state."""
    values = torch.linspace(
        -1.0,
        1.0,
        steps=FIXTURE_BATCH_SIZE * FIXTURE_INPUT_DIM * 2,
        dtype=torch.float32,
    ).reshape(2, FIXTURE_BATCH_SIZE, FIXTURE_INPUT_DIM)
    labels = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]],
        dtype=torch.long,
    )
    return (values[0], labels[0]), (values[1], labels[1])


def _training_step(
    torch: Any,
    adapter: Any,
    objective: Any,
    optimizer: Any,
    scaler: Any,
    batch: tuple[Any, Any],
    device: str,
) -> dict[str, Any]:
    """Apply the same fail-closed mixed-precision step as the epoch engine."""
    waveforms, labels = batch
    waveforms = waveforms.to(device)
    labels = labels.to(device)
    optimizer.zero_grad(set_to_none=True)
    scale_before = float(scaler.get_scale())
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        embeddings = adapter(waveforms)
        output = objective(embeddings, labels)
    if not bool(torch.isfinite(embeddings).all()) or not bool(
        torch.isfinite(output.loss)
    ):
        raise RuntimeError("Fixture embedding or loss became non-finite.")
    scaler.scale(output.loss).backward()
    scaler.unscale_(optimizer)
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
    if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("Fixture gradient norm became non-finite.")
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    if scale_after < scale_before:
        raise RuntimeError("Fixture GradScaler skipped the optimizer step.")
    return {
        "loss": float(output.loss.detach().cpu()),
        "accuracy": float(output.accuracy.detach().cpu()),
        "gradient_norm_before_clipping": float(gradient_norm.detach().cpu()),
        "loss_scale": scale_after,
        "embeddings": embeddings.detach(),
    }


def _component_hashes(
    torch: Any,
    adapter: Any,
    objective: Any,
    optimizer: Any,
    scaler: Any,
    embeddings: Any,
) -> dict[str, str]:
    """Fingerprint complete tensor/scalar state without retaining CPU copies."""
    return {
        "adapter": _hash_value(torch, adapter.state_dict()),
        "objective": _hash_value(torch, objective.state_dict()),
        "optimizer": _hash_value(torch, optimizer.state_dict()),
        "scaler": _hash_value(torch, scaler.state_dict()),
        "embeddings": _hash_value(torch, embeddings),
    }


def _hash_value(torch: Any, value: object) -> str:
    """Hash nested PyTorch state with explicit type and structure markers."""
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(tensor.numpy().tobytes(order="C"))
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(
                item,
                key=lambda candidate: (
                    type(candidate).__name__,
                    repr(candidate),
                ),
            ):
                update(key)
                update(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            for child in item:
                update(child)
            return
        if item is None or isinstance(item, (bool, int, float, str)):
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            digest.update(
                json.dumps(item, allow_nan=False, sort_keys=True).encode("utf-8")
            )
            digest.update(b"\0")
            return
        raise TypeError(f"Unsupported checkpoint hash value: {type(item)!r}")

    update(value)
    return digest.hexdigest()


def _serializable_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    """Remove the embedding tensor from a step record."""
    return {
        key: float(value)
        for key, value in values.items()
        if key != "embeddings"
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write finite gate evidence with deterministic formatting."""
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
