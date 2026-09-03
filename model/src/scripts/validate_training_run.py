"""Validate one completed training-run directory without loading PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from speaker_recognition.configuration import read_resolved_config


class TrainingRunEvidenceError(RuntimeError):
    """Raised when a run directory is incomplete or internally inconsistent."""


def parse_arguments() -> argparse.Namespace:
    """Parse the run directory produced by ``train_experiment.py``."""
    parser = argparse.ArgumentParser(
        description="Validate strict training, checkpoint, and Validation evidence."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def _reject_json_constant(value: str) -> None:
    """Reject non-standard NaN and infinity tokens during JSON parsing."""
    raise TrainingRunEvidenceError(f"Non-finite JSON constant: {value}.")


def _load_json(path: Path) -> Mapping[str, Any]:
    """Load one required strict JSON object."""
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingRunEvidenceError(f"Invalid JSON artifact: {path}") from error
    if not isinstance(payload, Mapping):
        raise TrainingRunEvidenceError(f"JSON root must be an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    """Hash one artifact without loading it completely into memory."""
    if not path.is_file():
        raise TrainingRunEvidenceError(f"Required artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    """Raise a bounded evidence error when one invariant is false."""
    if not condition:
        raise TrainingRunEvidenceError(message)


def _finite_number(value: object) -> bool:
    """Return whether a value is a finite non-boolean number."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def validate_training_run(run_directory: Path) -> Mapping[str, Any]:
    """Validate config, summary, checkpoints, and every Validation epoch."""
    root = run_directory.expanduser().resolve()
    resolved = read_resolved_config(root / "resolved_config.json")
    config = resolved.to_dict()
    summary = _load_json(root / "run_summary.json")
    identity = summary.get("identity")
    outcome = summary.get("outcome")
    dataset = summary.get("dataset")
    _require(summary.get("schema_version") == 1, "Run schema must equal 1.")
    _require(isinstance(identity, Mapping), "Run identity is missing.")
    _require(isinstance(outcome, Mapping), "Run outcome is missing.")
    _require(isinstance(dataset, Mapping), "Run dataset evidence is missing.")

    stage = config["experiment"]["stage"]
    model_name = config["model"]["name"]
    dataset_name = config["data"]["name"]
    _require(summary.get("stage") == stage, "Run stage differs from config.")
    _require(
        identity.get("config_sha256") == resolved.sha256,
        "Run config SHA-256 differs from the authenticated config.",
    )
    _require(identity.get("model_name") == model_name, "Model identity differs.")
    _require(
        identity.get("dataset_name") == dataset_name,
        "Dataset identity differs.",
    )
    _require(
        identity.get("seed") == config["experiment"]["seed"],
        "Run seed differs from config.",
    )
    manifest_fingerprint = identity.get("manifest_sha256")
    _require(
        isinstance(manifest_fingerprint, str)
        and len(manifest_fingerprint) == 64,
        "Manifest SHA-256 is invalid.",
    )

    history = outcome.get("history")
    memberships = dataset.get("epoch_memberships")
    _require(isinstance(history, list) and history, "Epoch history is empty.")
    _require(
        isinstance(memberships, list) and len(memberships) == len(history),
        "Epoch membership evidence differs from history length.",
    )
    max_epochs = config["training"]["lifecycle"]["max_epochs"]
    _require(len(history) <= max_epochs, "History exceeds configured epochs.")
    if stage == "pilot":
        _require(len(history) == 1, "Pilot must complete exactly one epoch.")

    expected_trial_sha256 = dataset.get("validation_trial_list_sha256")
    _require(
        isinstance(expected_trial_sha256, str)
        and len(expected_trial_sha256) == 64,
        "Validation trial SHA-256 is invalid.",
    )
    for epoch_index, (epoch, membership) in enumerate(
        zip(history, memberships, strict=True)
    ):
        _require(isinstance(epoch, Mapping), "Epoch history entry is invalid.")
        _require(
            isinstance(membership, Mapping),
            "Epoch membership entry is invalid.",
        )
        _require(epoch.get("epoch_index") == epoch_index, "Epoch order differs.")
        for field in (
            "training_loss",
            "training_accuracy",
            "validation_eer",
            "validation_min_dcf",
        ):
            _require(
                _finite_number(epoch.get(field)),
                f"Epoch {epoch_index} field {field} is not finite.",
            )
        _require(
            membership.get("epoch_index") == epoch_index,
            "Membership epoch index differs.",
        )
        membership_hash = membership.get("utterance_id_sha256")
        _require(
            isinstance(membership_hash, str) and len(membership_hash) == 64,
            "Epoch membership SHA-256 is invalid.",
        )
        if stage == "pilot":
            _require(
                membership.get("speaker_count") == 512
                and membership.get("utterance_count") == 512,
                "Pilot membership must contain 512 speakers and utterances.",
            )

        validation = _load_json(
            root / "validation" / f"validation_epoch_{epoch_index:04d}.json"
        )
        context = validation.get("context")
        protocol = validation.get("protocol")
        evaluation = validation.get("evaluation")
        metrics = validation.get("metrics")
        _require(isinstance(context, Mapping), "Validation context is missing.")
        _require(isinstance(protocol, Mapping), "Validation protocol is missing.")
        _require(isinstance(evaluation, Mapping), "Validation metadata is missing.")
        _require(isinstance(metrics, Mapping), "Validation metrics are missing.")
        _require(
            context.get("config_sha256") == resolved.sha256
            and context.get("manifest_sha256") == manifest_fingerprint,
            "Validation context fingerprints differ from the run.",
        )
        _require(
            protocol.get("trial_list_sha256") == expected_trial_sha256,
            "Validation trial fingerprint differs from the run.",
        )
        _require(
            evaluation.get("partition") == "validation"
            and evaluation.get("epoch_index") == epoch_index,
            "Validation partition or epoch differs.",
        )
        _require(
            all(_finite_number(value) for value in metrics.values()),
            "Validation metrics contain a non-finite value.",
        )

    checkpoint_hashes: dict[str, str] = {}
    for checkpoint_name in ("last.pt", "best.pt"):
        checkpoint = root / "checkpoints" / checkpoint_name
        sidecar = _load_json(checkpoint.with_name(checkpoint.name + ".json"))
        checkpoint_hashes[checkpoint_name] = _sha256_file(checkpoint)
        _require(
            sidecar.get("checkpoint_sha256") == checkpoint_hashes[checkpoint_name],
            f"{checkpoint_name} SHA-256 differs from its sidecar.",
        )
        _require(
            sidecar.get("identity") == identity,
            f"{checkpoint_name} identity differs from the run.",
        )

    if stage in {"resource_constrained", "full"}:
        final_summary = summary.get("final_test")
        final_test = _load_json(root / "final_test.json")
        final_context = final_test.get("context")
        final_evaluation = final_test.get("evaluation")
        final_protocol = final_test.get("protocol")
        final_selection = final_test.get("selection")
        final_threshold = final_test.get("threshold_policy")
        final_metrics = final_test.get("metrics")
        for name, value in (
            ("run summary", final_summary),
            ("context", final_context),
            ("evaluation", final_evaluation),
            ("protocol", final_protocol),
            ("selection", final_selection),
            ("threshold policy", final_threshold),
            ("metrics", final_metrics),
        ):
            _require(isinstance(value, Mapping), f"Final Test {name} is missing.")
        _require(
            final_evaluation.get("partition") == "test",
            "Final artifact must evaluate Test exactly once.",
        )
        _require(
            final_context.get("config_sha256") == resolved.sha256
            and final_context.get("manifest_sha256") == manifest_fingerprint,
            "Final Test context fingerprints differ from the run.",
        )
        _require(
            final_selection.get("checkpoint") == "checkpoints/best.pt"
            and final_selection.get("checkpoint_sha256")
            == checkpoint_hashes["best.pt"],
            "Final Test did not use the authenticated best checkpoint.",
        )
        best_epoch = final_selection.get("best_validation_epoch_index")
        _require(
            isinstance(best_epoch, int) and 0 <= best_epoch < len(history),
            "Final Test best Validation epoch is invalid.",
        )
        selected_validation = _load_json(
            root / "validation" / f"validation_epoch_{best_epoch:04d}.json"
        )
        selected_metrics = selected_validation.get("metrics")
        _require(
            isinstance(selected_metrics, Mapping)
            and final_selection.get("threshold_metric")
            == "threshold_at_far_0p1pct"
            and final_selection.get("frozen_threshold")
            == selected_metrics.get("threshold_at_far_0p1pct"),
            "Final Test threshold was not frozen from best Validation FAR 0.1%.",
        )
        _require(
            final_threshold.get("decision_threshold_source")
            == "frozen_validation_security_threshold"
            and final_threshold.get("security_threshold_selected") is True,
            "Final Test threshold provenance is invalid.",
        )
        _require(
            all(_finite_number(value) for value in final_metrics.values()),
            "Final Test metrics contain a non-finite value.",
        )
        _require(
            final_summary.get("checkpoint_sha256")
            == checkpoint_hashes["best.pt"]
            and final_summary.get("trial_list_sha256")
            == final_protocol.get("trial_list_sha256")
            and final_summary.get("metrics") == final_metrics,
            "Final Test summary differs from the immutable artifact.",
        )

    metrics_path = root / "metrics.jsonl"
    _require(metrics_path.is_file(), "Local JSONL metrics are missing.")
    _require(metrics_path.stat().st_size > 0, "Local JSONL metrics are empty.")
    return summary


def main() -> None:
    """Validate one run and print its accepted evidence identity."""
    arguments = parse_arguments()
    summary = validate_training_run(arguments.run_dir)
    print("TRAINING RUN EVIDENCE VALIDATED")
    print(f"run ID: {summary['run_id']}")
    print(f"stage: {summary['stage']}")
    print(f"epochs: {len(summary['outcome']['history'])}")
    print(f"run directory: {arguments.run_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
