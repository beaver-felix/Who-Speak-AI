"""Measure full forward/backward/optimizer memory on a real Kaggle T4 sample."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from speaker_recognition.configuration import resolve_layered_config
from speaker_recognition.data.audio import load_audio_file
from speaker_recognition.data.segments import evenly_spaced_segments
from speaker_recognition.training.specification import TrainingSpecification


TRAIN_SAMPLES = {
    "ecapa_tdnn": 48_000,
    "rawnet3": 48_240,
    "wavlm_mhfa": 48_240,
}


def parse_arguments() -> argparse.Namespace:
    """Parse one architecture, real sample, class count, and calibration grid."""
    parser = argparse.ArgumentParser(
        description="Calibrate mixed-precision training memory on Kaggle."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=tuple(TRAIN_SAMPLES),
    )
    parser.add_argument("--audio-file", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        required=True,
        type=int,
        help="Ascending candidate sizes, for example: 2 4 8 16.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Run one complete optimizer step per candidate and save evidence."""
    arguments = parse_arguments()
    _validate_arguments(arguments)

    # Delayed imports protect local dependency-free tests and Kaggle's
    # preinstalled CUDA-matched torch build.
    import torch

    from speaker_recognition.training.objectives import AamSoftmaxHead
    from speaker_recognition.training.optimization import build_optimizer

    if not torch.cuda.is_available() or not arguments.device.startswith("cuda"):
        raise RuntimeError("This calibration gate requires a CUDA device.")

    project_root = Path(__file__).resolve().parents[1]
    resolved = resolve_layered_config(
        (
            project_root / "configs/base.toml",
            project_root / "configs/datasets/vimd.toml",
            project_root / f"configs/models/{arguments.model}.toml",
        )
    )
    specification = TrainingSpecification.from_resolved_config(
        resolved.to_dict()
    )
    adapter = _load_adapter(arguments)
    head = AamSoftmaxHead(
        embedding_dim=adapter.metadata.embedding_dim,
        num_classes=arguments.num_classes,
        margin=specification.objective.margin,
        scale=specification.objective.scale,
        easy_margin=specification.objective.easy_margin,
    ).to(arguments.device)
    optimizer_bundle = build_optimizer(
        adapter,
        head,
        specification.optimization,
    )
    optimizer = optimizer_bundle.optimizer
    # A conservative initial scale avoids wasting early calibration attempts
    # at GradScaler's large general-purpose default. Dynamic backoff still
    # handles architecture-specific overflow safely.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
        init_scale=1024.0,
        growth_interval=100_000,
    )
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    adapter.train()
    head.train()

    audio = load_audio_file(arguments.audio_file, target_sample_rate=16000)
    crop = evenly_spaced_segments(
        audio.waveform,
        num_samples=TRAIN_SAMPLES[arguments.model],
        segment_count=1,
    )[0]
    base_waveform = torch.from_numpy(crop).to(arguments.device)

    measurements: list[dict[str, Any]] = []
    for batch_size in arguments.batch_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(arguments.device)
        try:
            # Deterministic circular shifts avoid a degenerate all-identical
            # batch while keeping this a reproducible single-record capacity
            # gate rather than a dataset-performance experiment.
            waveforms = torch.stack(
                [
                    torch.roll(base_waveform, shifts=index * 997)
                    for index in range(batch_size)
                ]
            )
            labels = torch.arange(
                batch_size,
                device=arguments.device,
                dtype=torch.long,
            ) % arguments.num_classes
            enabled_parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            attempts: list[dict[str, Any]] = []
            output = None
            gradient_norm = None
            optimizer_step_applied = False
            for attempt_index in range(8):
                optimizer.zero_grad(set_to_none=True)
                scale_before = float(scaler.get_scale())
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=True,
                ):
                    embeddings = adapter(waveforms)
                    output = head(embeddings, labels)
                scaler.scale(output.loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    enabled_parameters,
                    max_norm=specification.gradient_clip_norm,
                )
                loss_finite = bool(torch.isfinite(output.loss))
                gradient_norm_finite = bool(torch.isfinite(gradient_norm))
                head_before = (
                    head.weight[labels[0]].detach().clone()
                    if loss_finite and gradient_norm_finite
                    else None
                )
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                optimizer_step_applied = bool(
                    head_before is not None
                    and not torch.equal(
                        head_before,
                        head.weight[labels[0]].detach(),
                    )
                )
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "loss_finite": loss_finite,
                        "gradient_norm_finite": gradient_norm_finite,
                        "optimizer_step_applied": optimizer_step_applied,
                        "scale_before": scale_before,
                        "scale_after": scale_after,
                    }
                )
                if (
                    loss_finite
                    and gradient_norm_finite
                    and optimizer_step_applied
                ):
                    break

            torch.cuda.synchronize(arguments.device)
            free_bytes, total_bytes = torch.cuda.mem_get_info(arguments.device)
            passed = bool(
                output is not None
                and gradient_norm is not None
                and bool(torch.isfinite(output.loss))
                and bool(torch.isfinite(gradient_norm))
                and optimizer_step_applied
            )
            measurements.append(
                {
                    "batch_size": batch_size,
                    "status": "passed" if passed else "nonfinite_or_skipped",
                    "loss": (
                        float(output.loss.detach().cpu())
                        if output is not None and bool(torch.isfinite(output.loss))
                        else None
                    ),
                    "accuracy": (
                        float(output.accuracy.detach().cpu())
                        if output is not None
                        else None
                    ),
                    "gradient_norm_before_clipping": (
                        float(gradient_norm.detach().cpu())
                        if gradient_norm is not None
                        and bool(torch.isfinite(gradient_norm))
                        else None
                    ),
                    "loss_finite": bool(
                        output is not None and torch.isfinite(output.loss)
                    ),
                    "gradient_norm_finite": bool(
                        gradient_norm is not None
                        and torch.isfinite(gradient_norm)
                    ),
                    "optimizer_step_applied": optimizer_step_applied,
                    "attempts": attempts,
                    "peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(arguments.device)
                    ),
                    "peak_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(arguments.device)
                    ),
                    "device_free_bytes_after_step": int(free_bytes),
                    "device_total_bytes": int(total_bytes),
                }
            )
        except torch.OutOfMemoryError as error:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            measurements.append(
                {
                    "batch_size": batch_size,
                    "status": "out_of_memory",
                    "error": type(error).__name__,
                }
            )
            # Larger ascending candidates cannot fit after the first OOM.
            break

    passed_sizes = [
        item["batch_size"]
        for item in measurements
        if item["status"] == "passed"
    ]
    payload = {
        "schema_version": 2,
        "purpose": "memory_calibration_not_model_selection",
        "model": {
            "name": adapter.metadata.name,
            "embedding_dim": adapter.metadata.embedding_dim,
            "parameter_count": adapter.parameter_count,
        },
        "objective": {
            "name": "aam_softmax",
            "margin": specification.objective.margin,
            "scale": specification.objective.scale,
            "num_classes": arguments.num_classes,
        },
        "optimization": {
            "name": specification.optimization.optimizer,
            "group_names": list(optimizer_bundle.group_names),
            "group_parameter_counts": list(
                optimizer_bundle.group_parameter_counts
            ),
            "weight_decay": specification.optimization.weight_decay,
        },
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": arguments.device,
            "device_name": torch.cuda.get_device_name(arguments.device),
            "mixed_precision": "fp16",
        },
        "sample": {
            "path": str(arguments.audio_file),
            "crop_samples": TRAIN_SAMPLES[arguments.model],
        },
        "measurements": measurements,
        "largest_passing_batch_size": max(passed_sizes) if passed_sizes else None,
        "selection_rule": (
            "A passing size requires finite loss and gradient norm plus a "
            "verified optimizer update. Choose at most 80% of the largest "
            "passing size, then confirm with a multi-batch mini-run; "
            "calibration alone does not select a batch."
        ),
        "config_sha256": resolved.sha256,
    }
    _write_json(payload, arguments.output)
    if not passed_sizes:
        raise RuntimeError(
            f"{arguments.model} produced no finite applied optimizer step."
        )
    print("TRAINING MEMORY CALIBRATION COMPLETE")
    print(f"model: {arguments.model}")
    print(f"largest passing batch size: {payload['largest_passing_batch_size']}")
    print(f"output: {arguments.output.resolve()}")


def _load_adapter(arguments: argparse.Namespace) -> Any:
    """Construct one pinned model without importing the other optional stacks."""
    if arguments.model == "ecapa_tdnn":
        from speaker_recognition.models.ecapa_tdnn import EcapaTdnnAdapter

        return EcapaTdnnAdapter.from_pretrained(
            cache_dir=arguments.cache_dir,
            device=arguments.device,
        )
    if arguments.model == "rawnet3":
        from speaker_recognition.models.rawnet3 import RawNet3Adapter

        return RawNet3Adapter.from_pretrained(
            cache_dir=arguments.cache_dir,
            device=arguments.device,
        )

    from speaker_recognition.models.wavlm_mhfa import WavlmMhfaAdapter

    return WavlmMhfaAdapter.from_pretrained(
        cache_dir=arguments.cache_dir,
        device=arguments.device,
        checkpoint_path=arguments.checkpoint,
    )


def _validate_arguments(arguments: argparse.Namespace) -> None:
    """Reject unsafe or ambiguous calibration grids before model downloads."""
    if not arguments.audio_file.is_file():
        raise FileNotFoundError(arguments.audio_file)
    if arguments.num_classes <= 1:
        raise ValueError("num_classes must be greater than one.")
    if not arguments.batch_sizes or any(
        batch_size <= 0 for batch_size in arguments.batch_sizes
    ):
        raise ValueError("batch_sizes must contain positive integers.")
    if arguments.batch_sizes != sorted(set(arguments.batch_sizes)):
        raise ValueError("batch_sizes must be unique and strictly ascending.")
    if arguments.checkpoint is not None and arguments.model != "wavlm_mhfa":
        raise ValueError("--checkpoint is supported only for wavlm_mhfa.")


def _write_json(payload: dict[str, Any], output: Path) -> None:
    """Write deterministic structured evidence for later acceptance review."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
