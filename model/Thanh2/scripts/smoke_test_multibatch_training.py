"""Run three real multi-speaker optimizer steps on one pinned Kaggle model."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from speaker_recognition.configuration import resolve_layered_config
from speaker_recognition.data.dataset import (
    TrainingSpeakerDataset,
    collate_training_samples,
)
from speaker_recognition.data.manifest import ManifestRecord, Split
from speaker_recognition.data.tidyvoice import parse_tidyvoice_audio_path
from speaker_recognition.training.specification import TrainingSpecification


TRAIN_SAMPLES = {
    "ecapa_tdnn": 48_000,
    "rawnet3": 48_240,
    "wavlm_mhfa": 48_240,
}
TIDYVOICE_TRAIN_SPEAKERS = 3_666
DEFAULT_STEPS = 3


def parse_arguments() -> argparse.Namespace:
    """Parse one model, TidyVoice mount, cache, and evidence destination."""
    parser = argparse.ArgumentParser(
        description="Smoke-test real multi-batch speaker training on Kaggle."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=tuple(TRAIN_SAMPLES),
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Execute the accepted candidate batch through complete training steps."""
    arguments = parse_arguments()
    _validate_arguments(arguments)

    # Delayed import preserves local tests and Kaggle's CUDA-matched torch.
    import torch

    from speaker_recognition.training.objectives import AamSoftmaxHead
    from speaker_recognition.training.optimization import build_optimizer

    if not torch.cuda.is_available() or not arguments.device.startswith("cuda"):
        raise RuntimeError("This multi-batch gate requires a CUDA device.")

    project_root = Path(__file__).resolve().parents[1]
    resolved = resolve_layered_config(
        (
            project_root / "configs/base.toml",
            project_root / "configs/datasets/tidyvoice.toml",
            project_root / f"configs/models/{arguments.model}.toml",
        )
    )
    specification = TrainingSpecification.from_resolved_config(
        resolved.to_dict()
    )
    batch_size = specification.batch.size
    selected_count = batch_size * arguments.steps
    records, full_speaker_count = _select_distinct_speaker_records(
        arguments.dataset_root,
        count=selected_count,
    )
    if full_speaker_count != TIDYVOICE_TRAIN_SPEAKERS:
        raise RuntimeError(
            "TidyVoice training speaker count differs from the accepted audit: "
            f"expected {TIDYVOICE_TRAIN_SPEAKERS}, received "
            f"{full_speaker_count}."
        )

    dataset = TrainingSpeakerDataset(
        records,
        dataset_roots={"tidyvoice": arguments.dataset_root},
        segment_samples=TRAIN_SAMPLES[arguments.model],
        seed=42,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_training_samples,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False,
    )
    if len(data_loader) != arguments.steps:
        raise RuntimeError("Selected records did not form the requested batches.")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    adapter = _load_adapter(arguments)
    head = AamSoftmaxHead(
        embedding_dim=adapter.metadata.embedding_dim,
        num_classes=full_speaker_count,
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
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
        init_scale=1024.0,
        growth_interval=100_000,
    )
    adapter.train()
    head.train()

    steps: list[dict[str, Any]] = []
    observed_speakers: set[str] = set()
    observed_utterances: set[str] = set()
    observed_updated_groups: set[str] = set()
    mandatory_groups = (
        {"mhfa", "aam_softmax_head"}
        if arguments.model == "wavlm_mhfa"
        else set(optimizer_bundle.group_names)
    )
    for step_index, batch in enumerate(data_loader, start=1):
        if len(set(batch.speaker_ids)) != batch_size:
            raise RuntimeError("Every smoke batch must use distinct speakers.")
        if len(set(batch.utterance_ids)) != batch_size:
            raise RuntimeError("Every smoke batch must use distinct utterances.")
        if observed_speakers.intersection(batch.speaker_ids):
            raise RuntimeError("Speakers must not repeat across smoke batches.")
        observed_speakers.update(batch.speaker_ids)
        observed_utterances.update(batch.utterance_ids)

        waveforms = torch.from_numpy(batch.waveforms).to(arguments.device)
        labels = torch.from_numpy(batch.speaker_indices).to(arguments.device)
        if not bool(torch.isfinite(waveforms).all()):
            raise RuntimeError("Canonical training batch contains non-finite audio.")

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(arguments.device)
        started = time.perf_counter()
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
        enabled_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            enabled_parameters,
            max_norm=specification.gradient_clip_norm,
        )
        loss_finite = bool(torch.isfinite(output.loss))
        embeddings_finite = bool(torch.isfinite(embeddings).all())
        gradient_norm_finite = bool(torch.isfinite(gradient_norm))
        if not (loss_finite and embeddings_finite and gradient_norm_finite):
            raise RuntimeError(
                f"Non-finite training state at step {step_index}: "
                f"loss={loss_finite}, embeddings={embeddings_finite}, "
                f"gradient_norm={gradient_norm_finite}."
            )

        probes, groups_without_nonzero_gradient = _capture_group_probes(
            optimizer_bundle.group_names,
            optimizer.param_groups,
            torch,
        )
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(arguments.device)
        scale_after = float(scaler.get_scale())
        updated_groups = _updated_probe_groups(probes, torch)
        probed_groups = {probe[0] for probe in probes}
        if set(updated_groups) != probed_groups:
            raise RuntimeError(
                f"A probed optimizer group did not update at step {step_index}: "
                f"{updated_groups}."
            )
        if not mandatory_groups.issubset(updated_groups):
            raise RuntimeError(
                f"A mandatory optimizer group did not update at step "
                f"{step_index}: {updated_groups}."
            )
        observed_updated_groups.update(updated_groups)
        if scale_after < scale_before:
            raise RuntimeError(
                f"GradScaler skipped the optimizer step at batch {step_index}."
            )

        steps.append(
            {
                "step": step_index,
                "batch_size": batch_size,
                "distinct_speakers": len(set(batch.speaker_ids)),
                "distinct_utterances": len(set(batch.utterance_ids)),
                "loss": float(output.loss.detach().cpu()),
                "accuracy": float(output.accuracy.detach().cpu()),
                "gradient_norm_before_clipping": float(
                    gradient_norm.detach().cpu()
                ),
                "audio_finite": True,
                "embeddings_finite": embeddings_finite,
                "scale_before": scale_before,
                "scale_after": scale_after,
                "updated_groups": updated_groups,
                "groups_without_nonzero_gradient": (
                    groups_without_nonzero_gradient
                ),
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(arguments.device)
                ),
                "peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(arguments.device)
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )

    checks = {
        "step_count_matches": len(steps) == arguments.steps,
        "all_speakers_are_distinct": (
            len(observed_speakers) == selected_count
        ),
        "all_utterances_are_distinct": (
            len(observed_utterances) == selected_count
        ),
        "all_losses_finite": all(math_is_finite(step["loss"]) for step in steps),
        "all_audio_finite": all(step["audio_finite"] for step in steps),
        "all_embeddings_finite": all(
            step["embeddings_finite"] for step in steps
        ),
        "all_gradient_norms_finite": all(
            math_is_finite(step["gradient_norm_before_clipping"])
            for step in steps
        ),
        "all_available_groups_updated": all(
            not set(step["updated_groups"]).intersection(
                step["groups_without_nonzero_gradient"]
            )
            for step in steps
        ),
        "mandatory_groups_updated_each_step": all(
            mandatory_groups.issubset(step["updated_groups"])
            for step in steps
        ),
        "all_optimizer_groups_updated_across_gate": (
            observed_updated_groups == set(optimizer_bundle.group_names)
        ),
        "loss_scale_never_backed_off": all(
            step["scale_after"] >= step["scale_before"] for step in steps
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Multi-batch checks failed: {checks}")

    utterance_digest = hashlib.sha256(
        "\n".join(sorted(observed_utterances)).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "purpose": "multirecord_multibatch_training_gate_not_performance",
        "model": {
            "name": adapter.metadata.name,
            "embedding_dim": adapter.metadata.embedding_dim,
            "parameter_count": adapter.parameter_count,
        },
        "data": {
            "name": "tidyvoice",
            "full_training_speaker_count": full_speaker_count,
            "selected_speaker_count": len(observed_speakers),
            "selected_utterance_count": len(observed_utterances),
            "selected_utterance_ids_sha256": utterance_digest,
            "one_utterance_per_speaker": True,
            "segment_samples": TRAIN_SAMPLES[arguments.model],
        },
        "training": {
            "batch_size": batch_size,
            "steps": arguments.steps,
            "mixed_precision": "fp16",
            "gradient_clip_norm": specification.gradient_clip_norm,
            "objective": {
                "name": "aam_softmax",
                "margin": specification.objective.margin,
                "scale": specification.objective.scale,
            },
            "optimizer": {
                "name": specification.optimization.optimizer,
                "group_names": list(optimizer_bundle.group_names),
                "group_parameter_counts": list(
                    optimizer_bundle.group_parameter_counts
                ),
            },
        },
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": arguments.device,
            "device_name": torch.cuda.get_device_name(arguments.device),
        },
        "steps": steps,
        "checks": checks,
        "config_sha256": resolved.sha256,
    }
    _write_json(payload, arguments.output)
    print("REAL MULTI-BATCH TRAINING GATE PASSED")
    print(f"model: {arguments.model}")
    print(f"batch size: {batch_size}")
    print(f"steps: {arguments.steps}")
    print(f"distinct speakers: {len(observed_speakers)}")
    print(f"output: {arguments.output.resolve()}")


def _select_distinct_speaker_records(
    dataset_root: Path,
    *,
    count: int,
) -> tuple[tuple[ManifestRecord, ...], int]:
    """Select one deterministic real utterance from each earliest speaker."""
    branch_root = (
        dataset_root.expanduser().resolve()
        / "TidyVoiceX_Train"
        / "TidyVoiceX_Train"
    )
    if not branch_root.is_dir():
        raise FileNotFoundError(branch_root)
    speaker_directories = sorted(
        (path for path in branch_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    if len(speaker_directories) < count:
        raise RuntimeError(
            f"Need {count} distinct speakers, found {len(speaker_directories)}."
        )

    records: list[ManifestRecord] = []
    for speaker_directory in speaker_directories[:count]:
        candidates = sorted(
            (
                audio_path
                for language_directory in speaker_directory.iterdir()
                if language_directory.is_dir()
                for audio_path in language_directory.iterdir()
                if audio_path.is_file() and audio_path.suffix.lower() == ".wav"
            ),
            key=lambda path: path.relative_to(speaker_directory).as_posix(),
        )
        if not candidates:
            raise RuntimeError(
                f"Speaker has no WAV utterance: {speaker_directory.name}."
            )
        records.append(
            parse_tidyvoice_audio_path(
                candidates[0],
                dataset_root=dataset_root,
                split=Split.TRAIN,
            )
        )
    return tuple(records), len(speaker_directories)


def _capture_group_probes(
    group_names: Iterable[str],
    parameter_groups: Iterable[dict[str, Any]],
    torch: Any,
) -> tuple[list[tuple[str, Any, int, Any]], list[str]]:
    """Capture one update probe per active group and report inactive groups.

    WavLM's pinned encoder uses layerdrop 0.05 during training, so an individual
    Transformer group may intentionally have no gradient in one step. The gate
    requires every group to update across all steps instead of disabling this
    official regularization behavior.
    """
    probes: list[tuple[str, Any, int, Any]] = []
    inactive_groups: list[str] = []
    for name, group in zip(group_names, parameter_groups, strict=True):
        selected = None
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                continue
            nonzero = torch.nonzero(gradient.reshape(-1), as_tuple=False)
            if nonzero.numel() == 0:
                continue
            index = int(nonzero[0, 0])
            before = parameter.detach().reshape(-1)[index].clone()
            selected = (name, parameter, index, before)
            break
        if selected is None:
            inactive_groups.append(name)
        else:
            probes.append(selected)
    return probes, inactive_groups


def _updated_probe_groups(
    probes: Iterable[tuple[str, Any, int, Any]],
    torch: Any,
) -> list[str]:
    """Return group names whose selected scalar changed after optimizer step."""
    updated: list[str] = []
    for name, parameter, index, before in probes:
        after = parameter.detach().reshape(-1)[index]
        if not torch.equal(before, after):
            updated.append(name)
    return updated


def _load_adapter(arguments: argparse.Namespace) -> Any:
    """Construct only the selected pinned adapter and optional dependency stack."""
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
    """Reject invalid gates before model construction or audio decoding."""
    if not arguments.dataset_root.is_dir():
        raise FileNotFoundError(arguments.dataset_root)
    if arguments.steps < 2:
        raise ValueError("steps must be at least two for a multi-batch gate.")
    if arguments.checkpoint is not None and arguments.model != "wavlm_mhfa":
        raise ValueError("--checkpoint is supported only for wavlm_mhfa.")


def math_is_finite(value: object) -> bool:
    """Return whether a serialized numeric scalar is finite."""
    return isinstance(value, (int, float)) and np.isfinite(value).item()


def _write_json(payload: dict[str, Any], output: Path) -> None:
    """Write strict structured evidence; NaN and infinity fail serialization."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
