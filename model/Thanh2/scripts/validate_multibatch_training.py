"""Validate real multi-batch Kaggle evidence without importing PyTorch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


EXPECTED_MODELS = {
    "ecapa_tdnn": {
        "embedding_dim": 192,
        "parameter_count": 20_767_552,
        "batch_size": 24,
        "segment_samples": 48_000,
        "group_names": ("encoder", "aam_softmax_head"),
        "group_parameter_counts": (20_767_552, 703_872),
        "optimizer": "adam",
        "utterance_sha256": (
            "2de4dd8b8e25f8122f37ad8378e22abf28646ecc51816cfcd163e317dd381318"
        ),
        "config_sha256": (
            "df84325195e8aaa3f8c4fa55aeefd567fa299a9df70e8784c2a01a90efabbd39"
        ),
    },
    "rawnet3": {
        "embedding_dim": 256,
        "parameter_count": 16_280_322,
        "batch_size": 24,
        "segment_samples": 48_240,
        "group_names": ("encoder", "aam_softmax_head"),
        "group_parameter_counts": (16_280_322, 938_496),
        "optimizer": "adam",
        "utterance_sha256": (
            "2de4dd8b8e25f8122f37ad8378e22abf28646ecc51816cfcd163e317dd381318"
        ),
        "config_sha256": (
            "2b406d42e42759ad7b2ba2a590ba9cd990cf19cc864fc483bc0e6d9d9d568255"
        ),
    },
    "wavlm_mhfa": {
        "embedding_dim": 256,
        "parameter_count": 96_684_490,
        "batch_size": 6,
        "segment_samples": 48_240,
        "group_names": (
            *(f"transformer_layer_{index:02d}" for index in range(12)),
            "mhfa",
            "aam_softmax_head",
        ),
        "group_parameter_counts": (
            7_092_244,
            *(7_088_404 for _ in range(11)),
            2_302_554,
            938_496,
        ),
        "optimizer": "adamw",
        "utterance_sha256": (
            "4f01bd02c02ab81862723cfc181960927a31e73678db8ffc55910e8376bb211e"
        ),
        "config_sha256": (
            "7bcbd77b64f66a2b73c032e5dd321cbd4b0f4fe91290db926408a89448720534"
        ),
    },
}
EXPECTED_STEPS = 3
EXPECTED_CHECKS = {
    "step_count_matches",
    "all_speakers_are_distinct",
    "all_utterances_are_distinct",
    "all_losses_finite",
    "all_audio_finite",
    "all_embeddings_finite",
    "all_gradient_norms_finite",
    "all_available_groups_updated",
    "mandatory_groups_updated_each_step",
    "all_optimizer_groups_updated_across_gate",
    "loss_scale_never_backed_off",
}


def parse_arguments() -> argparse.Namespace:
    """Parse one or more downloaded multi-batch artifacts."""
    parser = argparse.ArgumentParser(
        description="Validate real multi-batch training evidence."
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    return parser.parse_args()


def validate_artifact(path: Path) -> dict[str, Any]:
    """Validate one artifact and return its compact acceptance summary."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError(f"{path.name}: schema_version must equal 1.")
    if payload.get("purpose") != (
        "multirecord_multibatch_training_gate_not_performance"
    ):
        raise ValueError(f"{path.name}: invalid evidence purpose.")

    model = _mapping(payload, "model", path)
    model_name = str(model.get("name"))
    expected = EXPECTED_MODELS.get(model_name)
    if expected is None:
        raise ValueError(f"{path.name}: unsupported model {model_name!r}.")
    for field in ("embedding_dim", "parameter_count"):
        if model.get(field) != expected[field]:
            raise ValueError(f"{path.name}: invalid model {field}.")

    data = _mapping(payload, "data", path)
    training = _mapping(payload, "training", path)
    optimizer = _mapping(training, "optimizer", path)
    batch_size = expected["batch_size"]
    selected_count = int(batch_size) * EXPECTED_STEPS
    expected_data = {
        "name": "tidyvoice",
        "full_training_speaker_count": 3_666,
        "selected_speaker_count": selected_count,
        "selected_utterance_count": selected_count,
        "one_utterance_per_speaker": True,
        "segment_samples": expected["segment_samples"],
    }
    for field, expected_value in expected_data.items():
        if data.get(field) != expected_value:
            raise ValueError(f"{path.name}: invalid data {field}.")
    digest = data.get("selected_utterance_ids_sha256")
    if digest != expected["utterance_sha256"]:
        raise ValueError(f"{path.name}: invalid utterance fingerprint.")
    if training.get("batch_size") != batch_size:
        raise ValueError(f"{path.name}: invalid candidate batch size.")
    if training.get("steps") != EXPECTED_STEPS:
        raise ValueError(f"{path.name}: expected three training steps.")
    if training.get("mixed_precision") != "fp16":
        raise ValueError(f"{path.name}: mixed precision must be fp16.")
    expected_groups = tuple(expected["group_names"])
    if tuple(optimizer.get("group_names", ())) != expected_groups:
        raise ValueError(f"{path.name}: optimizer groups changed.")
    if tuple(optimizer.get("group_parameter_counts", ())) != tuple(
        expected["group_parameter_counts"]
    ):
        raise ValueError(f"{path.name}: optimizer group sizes changed.")
    if optimizer.get("name") != expected["optimizer"]:
        raise ValueError(f"{path.name}: optimizer changed.")
    expected_objective = {
        "name": "aam_softmax",
        "margin": 0.2,
        "scale": 30.0,
    }
    if training.get("objective") != expected_objective:
        raise ValueError(f"{path.name}: objective changed.")

    checks = _mapping(payload, "checks", path)
    if set(checks) != EXPECTED_CHECKS or not all(
        checks.get(name) is True for name in EXPECTED_CHECKS
    ):
        raise ValueError(f"{path.name}: one or more required checks failed.")
    steps = payload.get("steps")
    if not isinstance(steps, list) or len(steps) != EXPECTED_STEPS:
        raise ValueError(f"{path.name}: invalid step records.")
    mandatory_groups = (
        {"mhfa", "aam_softmax_head"}
        if model_name == "wavlm_mhfa"
        else set(expected_groups)
    )
    observed_updated_groups: set[str] = set()
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping) or step.get("step") != index:
            raise ValueError(f"{path.name}: malformed step {index}.")
        if step.get("batch_size") != batch_size:
            raise ValueError(f"{path.name}: step batch size changed.")
        if step.get("distinct_speakers") != batch_size:
            raise ValueError(f"{path.name}: repeated speaker in step {index}.")
        if step.get("distinct_utterances") != batch_size:
            raise ValueError(f"{path.name}: repeated utterance in step {index}.")
        if step.get("audio_finite") is not True:
            raise ValueError(f"{path.name}: non-finite audio in step {index}.")
        if step.get("embeddings_finite") is not True:
            raise ValueError(f"{path.name}: non-finite embeddings in step {index}.")
        for field in (
            "loss",
            "accuracy",
            "gradient_norm_before_clipping",
            "scale_before",
            "scale_after",
            "elapsed_seconds",
        ):
            if not _finite_number(step.get(field)):
                raise ValueError(f"{path.name}: non-finite {field} in step {index}.")
        if float(step["scale_after"]) < float(step["scale_before"]):
            raise ValueError(f"{path.name}: loss-scale backoff in step {index}.")
        updated_groups = tuple(step.get("updated_groups", ()))
        inactive_groups = tuple(step.get("groups_without_nonzero_gradient", ()))
        if len(updated_groups) != len(set(updated_groups)):
            raise ValueError(f"{path.name}: duplicate update group in step {index}.")
        if not set(updated_groups).issubset(expected_groups):
            raise ValueError(f"{path.name}: unknown update group in step {index}.")
        if set(updated_groups).intersection(inactive_groups):
            raise ValueError(f"{path.name}: inconsistent inactive group in step {index}.")
        if set(updated_groups).union(inactive_groups) != set(expected_groups):
            raise ValueError(f"{path.name}: incomplete group accounting in step {index}.")
        if not mandatory_groups.issubset(updated_groups):
            raise ValueError(f"{path.name}: mandatory group missed step {index}.")
        observed_updated_groups.update(updated_groups)
        for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
            if not _positive_integer(step.get(field)):
                raise ValueError(f"{path.name}: invalid {field} in step {index}.")

    runtime = _mapping(payload, "runtime", path)
    if observed_updated_groups != set(expected_groups):
        raise ValueError(f"{path.name}: not every group updated across the gate.")
    if runtime.get("device_name") != "Tesla T4":
        raise ValueError(f"{path.name}: evidence must come from a Tesla T4.")
    if runtime.get("torch_version") != "2.10.0+cu128":
        raise ValueError(f"{path.name}: unexpected PyTorch runtime.")
    if runtime.get("cuda_version") != "12.8":
        raise ValueError(f"{path.name}: unexpected CUDA runtime.")
    config_sha256 = payload.get("config_sha256")
    if config_sha256 != expected["config_sha256"]:
        raise ValueError(f"{path.name}: invalid configuration fingerprint.")
    return {
        "model": model_name,
        "batch_size": batch_size,
        "steps": EXPECTED_STEPS,
        "distinct_speakers": selected_count,
    }


def _mapping(
    values: Mapping[str, Any],
    key: str,
    path: Path,
) -> Mapping[str, Any]:
    """Read one required nested mapping with artifact-specific context."""
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name}: {key} must be a mapping.")
    return value


def _finite_number(value: object) -> bool:
    """Return whether a JSON value is numeric, non-boolean, and finite."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _positive_integer(value: object) -> bool:
    """Return whether a JSON value is a strict positive integer."""
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def main() -> None:
    """Validate each model once and print compact accepted summaries."""
    arguments = parse_arguments()
    seen_models: set[str] = set()
    for artifact in arguments.artifacts:
        summary = validate_artifact(artifact.resolve())
        model_name = str(summary["model"])
        if model_name in seen_models:
            raise ValueError(f"Duplicate model evidence: {model_name}.")
        seen_models.add(model_name)
        print(
            f"{model_name}: batch={summary['batch_size']}, "
            f"steps={summary['steps']}, "
            f"distinct_speakers={summary['distinct_speakers']}"
        )
    print("REAL MULTI-BATCH TRAINING EVIDENCE VALIDATED")


if __name__ == "__main__":
    main()
