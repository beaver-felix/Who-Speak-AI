"""Run one restart-safe model/dataset speaker-recognition experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from speaker_recognition.configuration import read_resolved_config
from speaker_recognition.data.manifest import (
    ManifestRecord,
    Split,
    manifest_sha256,
    validate_manifest,
)
from speaker_recognition.data.tidyvoice import iter_tidyvoice_manifest_records
from speaker_recognition.data.vimd import iter_vimd_source_records
from speaker_recognition.data.sampling import utterance_id_sha256
from speaker_recognition.evaluation.trials import (
    build_verification_trials,
    trial_list_sha256,
)
from speaker_recognition.training.specification import TrainingSpecification


class ExperimentAssemblyError(RuntimeError):
    """Raised when a declared experiment cannot be assembled exactly."""


def parse_arguments() -> argparse.Namespace:
    """Parse one immutable resolved configuration and runtime directories."""
    parser = argparse.ArgumentParser(
        description="Train one of the six canonical speaker experiments."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume strictly from run-dir/checkpoints/last.pt.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Mapping[str, object]:
    """Load one required JSON object."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentAssemblyError(f"Unable to read JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise ExperimentAssemblyError(f"JSON root must be an object: {path}")
    return payload


def _load_tidyvoice_assignments(path: Path) -> dict[str, Split]:
    """Read the accepted speaker-disjoint TidyVoice Dev assignment."""
    payload = _load_json(path)
    values = payload.get("assignments")
    if not isinstance(values, Mapping) or not values:
        raise ExperimentAssemblyError(
            "TidyVoice protocol assignments are missing."
        )
    try:
        return {
            str(speaker_id): Split(split_name)
            for speaker_id, split_name in values.items()
        }
    except ValueError as error:
        raise ExperimentAssemblyError(
            "TidyVoice protocol contains an invalid split."
        ) from error


def _build_manifest(
    config: Mapping[str, Any],
    *,
    dataset_root: Path,
    project_root: Path,
) -> tuple[ManifestRecord, ...]:
    """Build and verify one complete canonical dataset manifest."""
    data = config["data"]
    dataset_name = str(data["name"])
    if dataset_name == "tidyvoice":
        protocol_path = project_root / str(data["split_protocol"])
        assignments = _load_tidyvoice_assignments(protocol_path)
        records = tuple(
            iter_tidyvoice_manifest_records(
                dataset_root,
                dev_assignments=assignments,
            )
        )
    elif dataset_name == "vimd":
        batch_size = int(data["parquet"]["metadata_batch_size"])
        records = tuple(
            record
            for source_split in ("train", "valid", "test")
            for record in iter_vimd_source_records(
                dataset_root,
                source_split=source_split,
                batch_size=batch_size,
            )
        )
    else:
        raise ExperimentAssemblyError(
            f"Unsupported canonical dataset: {dataset_name!r}."
        )
    validated = validate_manifest(records)
    observed_counts = Counter(record.split.value for record in validated)
    declared_counts = data["canonical_counts"]
    expected_counts = {
        "train": int(declared_counts["train_utterances"]),
        "validation": int(declared_counts["validation_utterances"]),
        "test": int(declared_counts["test_utterances"]),
    }
    if dict(observed_counts) != expected_counts:
        raise ExperimentAssemblyError(
            "Canonical manifest counts differ from the resolved configuration: "
            f"expected {expected_counts}, observed {dict(observed_counts)}."
        )
    return validated


def _build_adapter(
    model_name: str,
    *,
    cache_dir: Path,
    device: str,
) -> Any:
    """Load one pinned adapter without importing unused model dependencies."""
    if model_name == "ecapa_tdnn":
        from speaker_recognition.models.ecapa_tdnn import EcapaTdnnAdapter

        return EcapaTdnnAdapter.from_pretrained(
            cache_dir=cache_dir,
            device=device,
        )
    if model_name == "rawnet3":
        from speaker_recognition.models.rawnet3 import RawNet3Adapter

        return RawNet3Adapter.from_pretrained(
            cache_dir=cache_dir,
            device=device,
        )
    if model_name == "wavlm_mhfa":
        from speaker_recognition.models.wavlm_mhfa import WavlmMhfaAdapter

        return WavlmMhfaAdapter.from_pretrained(
            cache_dir=cache_dir,
            device=device,
        )
    raise ExperimentAssemblyError(f"Unsupported model: {model_name!r}.")


def _expected_trial_sha256(
    project_root: Path,
    *,
    dataset_name: str,
) -> str:
    """Read the immutable canonical Validation trial fingerprint."""
    payload = _load_json(
        project_root / "results/data_audit/verification_trial_protocols.json"
    )
    try:
        value = payload["protocols"][dataset_name]["validation"][
            "trial_list_sha256"
        ]
    except (KeyError, TypeError) as error:
        raise ExperimentAssemblyError(
            "Canonical Validation trial fingerprint is missing."
        ) from error
    if not isinstance(value, str) or len(value) != 64:
        raise ExperimentAssemblyError(
            "Canonical Validation trial fingerprint is invalid."
        )
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write strict result JSON through atomic sibling replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _stable_run_id(
    *,
    dataset_name: str,
    model_name: str,
    seed: int,
    config_sha256: str,
) -> str:
    """Build one explicit W&B identity shared across interrupted sessions."""
    payload = f"{dataset_name}\0{model_name}\0{seed}\0{config_sha256}".encode(
        "utf-8"
    )
    suffix = hashlib.sha256(payload).hexdigest()[:12]
    return f"{dataset_name}-{model_name}-s{seed}-{suffix}"


def main() -> None:
    """Assemble and run one immutable pilot or full training experiment."""
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    resolved = read_resolved_config(arguments.config)
    config = resolved.to_dict()
    stage = str(config["experiment"]["stage"])
    if stage not in {"pilot", "full"}:
        raise ExperimentAssemblyError(
            "Training requires an explicit pilot or full stage layer."
        )

    reproducibility = config["reproducibility"]
    expected_workspace = str(reproducibility["cublas_workspace_config"])
    existing_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_workspace not in {None, expected_workspace}:
        raise ExperimentAssemblyError(
            "CUBLAS_WORKSPACE_CONFIG conflicts with the resolved config."
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected_workspace

    # All CUDA imports occur after the deterministic cuBLAS environment is set.
    import torch

    from speaker_recognition.data.dataset import (
        EvaluationSpeakerDataset,
        TrainingSpeakerDataset,
    )
    from speaker_recognition.evaluation.runtime import ExtractionSettings
    from speaker_recognition.evaluation.validation import (
        VerificationValidationCallback,
    )
    from speaker_recognition.training.engine import (
        SpeakerTrainingEngine,
        TrainerSettings,
    )
    from speaker_recognition.training.lifecycle import (
        EarlyStoppingPolicy,
        TrainingRunIdentity,
    )
    from speaker_recognition.training.logging import (
        CompositeRunLogger,
        JsonlRunLogger,
        WandbRunLogger,
    )
    from speaker_recognition.training.objectives import AamSoftmaxHead
    from speaker_recognition.training.optimization import build_optimizer

    if not torch.cuda.is_available() or not arguments.device.startswith("cuda"):
        raise ExperimentAssemblyError("Accepted training requires CUDA.")
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(
        reproducibility["cudnn_benchmark"]
    )
    torch.use_deterministic_algorithms(
        bool(reproducibility["deterministic_algorithms"])
    )

    specification = TrainingSpecification.from_resolved_config(config)
    data = config["data"]
    model = config["model"]
    dataset_name = str(data["name"])
    model_name = str(model["name"])
    dataset_root = (
        arguments.dataset_root
        if arguments.dataset_root is not None
        else Path(str(data["kaggle_root"]))
    ).expanduser().resolve()
    if not dataset_root.is_dir():
        raise ExperimentAssemblyError(
            f"Dataset root does not exist: {dataset_root}"
        )
    run_directory = arguments.run_dir.expanduser().resolve()
    checkpoints = run_directory / "checkpoints"
    last_checkpoint = checkpoints / "last.pt"
    best_checkpoint = checkpoints / "best.pt"
    metrics_path = run_directory / "metrics.jsonl"
    if arguments.resume and not last_checkpoint.is_file():
        raise ExperimentAssemblyError(
            "--resume requires run-dir/checkpoints/last.pt."
        )
    if not arguments.resume and (
        last_checkpoint.exists() or metrics_path.exists()
    ):
        raise ExperimentAssemblyError(
            "Existing run state requires --resume or a new run directory."
        )
    _write_json(
        run_directory / "resolved_config.json",
        {
            "config": config,
            "config_sha256": resolved.sha256,
            "source_layers": list(resolved.source_paths),
        },
    )

    print("ASSEMBLING CANONICAL EXPERIMENT")
    print(f"stage: {stage}")
    print(f"dataset: {dataset_name}")
    print(f"model: {model_name}")
    print(f"config SHA-256: {resolved.sha256}")

    manifest = _build_manifest(
        config,
        dataset_root=dataset_root,
        project_root=project_root,
    )
    manifest_fingerprint = manifest_sha256(manifest)
    roots = {dataset_name: dataset_root}
    training_dataset = TrainingSpeakerDataset(
        manifest,
        dataset_roots=roots,
        segment_samples=specification.training_segment_samples,
        seed=seed,
        max_utterances_per_speaker=(
            specification.epoch_sampling.max_utterances_per_speaker
        ),
        max_speakers_per_epoch=(
            specification.epoch_sampling.max_speakers_per_epoch
        ),
    )

    verification = config["verification"]
    validation_trials = build_verification_trials(
        manifest,
        split=Split.VALIDATION,
        seed=int(verification["seed"]),
        max_genuine_per_speaker=int(
            verification["max_genuine_per_speaker"]
        ),
        impostor_trial_count=int(
            verification["impostor_trials_per_split"]
        ),
    )
    expected_trial_fingerprint = _expected_trial_sha256(
        project_root,
        dataset_name=dataset_name,
    )
    if trial_list_sha256(validation_trials) != expected_trial_fingerprint:
        raise ExperimentAssemblyError(
            "Regenerated Validation trials differ from the committed protocol."
        )
    validation_ids = {
        utterance_id
        for trial in validation_trials
        for utterance_id in (
            trial.left_utterance_id,
            trial.right_utterance_id,
        )
    }
    validation_records = tuple(
        record
        for record in manifest
        if record.utterance_id in validation_ids
    )
    evaluation = specification.evaluation
    validation_dataset = EvaluationSpeakerDataset(
        validation_records,
        split=Split.VALIDATION,
        dataset_roots=roots,
        segment_samples=evaluation.segment_samples,
        segment_count=evaluation.segment_count,
    )

    adapter = _build_adapter(
        model_name,
        cache_dir=arguments.cache_dir.expanduser().resolve(),
        device=arguments.device,
    )
    objective_spec = specification.objective
    objective = AamSoftmaxHead(
        embedding_dim=int(model["embedding_dim"]),
        num_classes=len(training_dataset.speaker_to_index),
        margin=objective_spec.margin,
        scale=objective_spec.scale,
        easy_margin=objective_spec.easy_margin,
    ).to(arguments.device)
    optimizer_bundle = build_optimizer(
        adapter,
        objective,
        specification.optimization,
    )
    identity = TrainingRunIdentity(
        model_name=model_name,
        dataset_name=dataset_name,
        config_sha256=resolved.sha256,
        manifest_sha256=manifest_fingerprint,
        seed=seed,
    )
    lifecycle = specification.lifecycle
    engine = SpeakerTrainingEngine(
        adapter=adapter,
        objective=objective,
        optimizer_bundle=optimizer_bundle,
        dataset=training_dataset,
        identity=identity,
        settings=TrainerSettings(
            batch_size=specification.batch.size,
            max_epochs=lifecycle.max_epochs,
            gradient_clip_norm=specification.gradient_clip_norm,
            checkpoint_every_steps=lifecycle.checkpoint_every_steps,
            num_workers=lifecycle.num_workers,
            pin_memory=lifecycle.pin_memory,
            initial_loss_scale=lifecycle.initial_loss_scale,
            group_by_audio_path=specification.group_by_audio_path,
        ),
        early_stopping=EarlyStoppingPolicy(
            patience=lifecycle.early_stopping_patience,
            minimum_eer_improvement=lifecycle.minimum_eer_improvement,
        ),
        device=arguments.device,
    )
    if arguments.resume:
        restored_sha256 = engine.resume_from(last_checkpoint)
        print(f"RESUMED CHECKPOINT SHA-256: {restored_sha256}")

    validation_callback = VerificationValidationCallback(
        dataset=validation_dataset,
        trials=validation_trials,
        expected_trial_sha256=expected_trial_fingerprint,
        extraction_settings=ExtractionSettings(
            utterance_batch_size=evaluation.utterance_batch_size,
            num_workers=evaluation.num_workers,
            pin_memory=evaluation.pin_memory,
        ),
        output_directory=run_directory / "validation",
        evidence_context={
            "stage": stage,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "config_sha256": resolved.sha256,
            "manifest_sha256": manifest_fingerprint,
            "seed": seed,
        },
        device=arguments.device,
    )

    run_id = _stable_run_id(
        dataset_name=dataset_name,
        model_name=model_name,
        seed=seed,
        config_sha256=resolved.sha256,
    )
    local_logger = JsonlRunLogger(
        metrics_path,
        resume_step=engine.cursor.global_step if arguments.resume else None,
    )
    tracking = config["tracking"]
    wandb_logger = WandbRunLogger(
        project=str(tracking["project"]),
        run_id=run_id,
        run_name=run_id,
        config=config,
        directory=run_directory / "wandb",
        resume=arguments.resume,
        resume_step=(engine.cursor.global_step if arguments.resume else None),
        mode=str(tracking["mode"]),
    )
    outcome = engine.fit(
        validation_function=validation_callback,
        last_checkpoint_path=last_checkpoint,
        best_checkpoint_path=best_checkpoint,
        logger=CompositeRunLogger((local_logger, wandb_logger)),
    )

    epoch_memberships = []
    for completed_epoch in range(len(outcome.history)):
        training_dataset.set_epoch(completed_epoch)
        epoch_memberships.append(
            {
                "epoch_index": completed_epoch,
                "speaker_count": len(
                    {
                        record.speaker_id
                        for record in training_dataset.epoch_records
                    }
                ),
                "utterance_count": len(training_dataset.epoch_records),
                "utterance_id_sha256": utterance_id_sha256(
                    training_dataset.epoch_records
                ),
            }
        )

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": stage,
        "identity": identity.to_dict(),
        "dataset": {
            "canonical_utterance_count": len(manifest),
            "training_speaker_count": len(training_dataset.speaker_to_index),
            "epoch_utterance_count": len(training_dataset),
            "validation_trial_utterance_count": len(validation_dataset),
            "validation_trial_count": len(validation_trials),
            "validation_trial_list_sha256": expected_trial_fingerprint,
            "epoch_memberships": epoch_memberships,
        },
        "outcome": {
            "cursor": outcome.cursor.to_dict(),
            "history": [entry.to_dict() for entry in outcome.history],
            "stopped_early": outcome.stopped_early,
        },
        "runtime": {
            "device": arguments.device,
            "device_name": torch.cuda.get_device_name(arguments.device),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
        },
    }
    _write_json(run_directory / "run_summary.json", summary)
    print("TRAINING EXPERIMENT COMPLETE")
    print(f"run ID: {run_id}")
    print(f"stage: {stage}")
    print(f"epochs completed: {len(outcome.history)}")
    print(f"best epoch: {outcome.cursor.best_epoch_index}")
    print(f"best Validation EER: {outcome.cursor.best_validation_eer}")
    print(f"summary: {(run_directory / 'run_summary.json').resolve()}")


if __name__ == "__main__":
    main()
