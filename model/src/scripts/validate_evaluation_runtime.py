"""Validate real CUDA evaluation evidence without importing model stacks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping


EXPECTED_MODELS = {
    "ecapa_tdnn": (192, 48_000, 4),
    "rawnet3": (256, 64_240, 4),
    "wavlm_mhfa": (256, 64_240, 2),
}
EXPECTED_CHECKS = {
    "trial_fingerprint_matches",
    "exact_trial_count",
    "exact_genuine_count",
    "exact_impostor_count",
    "exact_utterance_count",
    "valid_variable_crop_coverage",
    "complete_required_metrics",
    "metrics_finite",
    "security_threshold_not_selected",
    "returned_metrics_finite",
    "latency_finite_positive",
}
REQUIRED_METRICS = {
    "eer",
    "min_dcf",
    "far",
    "frr",
    "tar",
    "accuracy",
    "tar_at_far_5pct",
    "tar_at_far_1pct",
    "tar_at_far_0p1pct",
    "tar_at_far_0p01pct",
}


def parse_arguments() -> argparse.Namespace:
    """Parse one or more downloaded evaluation-gate artifacts."""
    parser = argparse.ArgumentParser(
        description="Validate bounded real-audio evaluation runtime evidence."
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    return parser.parse_args()


def validate_artifact(path: Path) -> dict[str, object]:
    """Reject an incomplete, changed, non-finite, or non-T4 gate artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("Evaluation evidence schema_version must equal 1.")
    if payload.get("purpose") != "bounded_real_audio_evaluation_runtime_gate":
        raise ValueError("Evaluation evidence purpose changed.")

    model = _mapping(payload, "model")
    model_name = model.get("name")
    if not isinstance(model_name, str) or model_name not in EXPECTED_MODELS:
        raise ValueError("Evaluation evidence model is unsupported.")
    embedding_dim, segment_samples, batch_size = EXPECTED_MODELS[model_name]
    if model.get("embedding_dim") != embedding_dim:
        raise ValueError("Model embedding dimension changed.")
    if not _is_commit_sha(model.get("revision")):
        raise ValueError("Model revision must be a pinned 40-character commit.")

    fixture = _mapping(payload, "fixture")
    expected_fixture = {
        "dataset": "tidyvoice",
        "partition": "validation",
        "speaker_count": 4,
        "utterance_count": 8,
        "trial_count": 16,
        "genuine_trial_count": 4,
        "impostor_trial_count": 12,
        "segment_samples": segment_samples,
        "segment_count": 2,
        "utterance_batch_size": batch_size,
    }
    for key, expected in expected_fixture.items():
        if fixture.get(key) != expected:
            raise ValueError(f"Evaluation fixture {key} changed.")
    if not _is_sha256(fixture.get("trial_list_sha256")):
        raise ValueError("Fixture trial SHA-256 is invalid.")

    runtime = _mapping(payload, "runtime")
    if runtime.get("device_name") != "Tesla T4":
        raise ValueError("Evaluation evidence must come from a Tesla T4.")
    if runtime.get("device") != "cuda:0":
        raise ValueError("Evaluation evidence must use cuda:0.")
    if runtime.get("torch_version") != "2.10.0+cu128":
        raise ValueError("Evaluation evidence used an unexpected PyTorch.")
    if runtime.get("cuda_version") != "12.8":
        raise ValueError("Evaluation evidence used an unexpected CUDA.")
    if not _positive_number(runtime.get("maximum_allocated_cuda_bytes")):
        raise ValueError("Evaluation CUDA memory must be positive.")

    validation = _mapping(payload, "validation")
    protocol = _mapping(validation, "protocol")
    embeddings = _mapping(validation, "embeddings")
    threshold = _mapping(validation, "threshold_policy")
    metrics = _mapping(validation, "metrics")
    evaluation = _mapping(validation, "evaluation")
    extraction = _mapping(evaluation, "embedding_extraction")
    if protocol.get("trial_list_sha256") != fixture["trial_list_sha256"]:
        raise ValueError("Validation trial fingerprint changed.")
    if protocol.get("trial_count") != 16:
        raise ValueError("Validation trial count changed.")
    if embeddings.get("utterance_count") != 8:
        raise ValueError("Validation utterance count changed.")
    if not _is_sha256(embeddings.get("embedding_table_sha256")):
        raise ValueError("Embedding-table SHA-256 is invalid.")
    if threshold.get("security_threshold_selected") is not False:
        raise ValueError("The bounded gate cannot select a security threshold.")
    if REQUIRED_METRICS - set(metrics):
        raise ValueError("Required verification metrics are incomplete.")
    if not all(_finite_number(value) for value in metrics.values()):
        raise ValueError("Verification metrics must all be finite.")
    if not 0.0 <= float(metrics["eer"]) <= 1.0:
        raise ValueError("EER must be in [0, 1].")
    if float(metrics["min_dcf"]) < 0.0:
        raise ValueError("minDCF must be non-negative.")
    if extraction.get("utterance_count") != 8:
        raise ValueError("Extraction utterance coverage changed.")
    crop_count = extraction.get("crop_count")
    if (
        isinstance(crop_count, bool)
        or not isinstance(crop_count, int)
        or not 8 <= crop_count <= 16
    ):
        raise ValueError(
            "Extraction must contain one or two crops per utterance."
        )
    if not all(_positive_number(value) for value in extraction.values()):
        raise ValueError("Extraction statistics must all be positive.")

    latency = _mapping(payload, "latency")
    if latency.get("warmup_iterations") != 10:
        raise ValueError("Latency warm-up count changed.")
    if latency.get("measured_iterations") != 50:
        raise ValueError("Latency measurement count changed.")
    if not all(
        _positive_number(latency.get(field))
        for field in ("mean_ms", "median_ms", "p95_ms")
    ):
        raise ValueError("Latency values must be finite and positive.")

    checks = _mapping(payload, "checks")
    if set(checks) != EXPECTED_CHECKS or not all(
        checks.get(name) is True for name in EXPECTED_CHECKS
    ):
        raise ValueError("One or more evaluation runtime checks failed.")
    return {
        "model": model_name,
        "eer": float(metrics["eer"]),
        "min_dcf": float(metrics["min_dcf"]),
        "median_latency_ms": float(latency["median_ms"]),
    }


def _mapping(values: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Read one required nested mapping."""
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Evaluation evidence {key} must be a mapping.")
    return value


def _finite_number(value: object) -> bool:
    """Return whether a value is numeric, non-boolean, and finite."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _positive_number(value: object) -> bool:
    """Return whether a value is finite and strictly positive."""
    return _finite_number(value) and float(value) > 0.0


def _is_sha256(value: object) -> bool:
    """Return whether a value is a lowercase hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and value.lower() == value
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit_sha(value: object) -> bool:
    """Return whether a value is a lowercase 40-character Git commit."""
    return (
        isinstance(value, str)
        and len(value) == 40
        and value.lower() == value
        and all(character in "0123456789abcdef" for character in value)
    )


def main() -> None:
    """Validate every supplied artifact and print compact accepted results."""
    arguments = parse_arguments()
    summaries = [validate_artifact(path) for path in arguments.artifacts]
    model_names = [str(summary["model"]) for summary in summaries]
    if len(set(model_names)) != len(model_names):
        raise ValueError("Evaluation artifacts contain duplicate model names.")
    for summary in summaries:
        print(
            "EVALUATION RUNTIME EVIDENCE VALIDATED: "
            f"{summary['model']}, EER={summary['eer']:.6f}, "
            f"minDCF={summary['min_dcf']:.6f}, "
            f"median={summary['median_latency_ms']:.3f} ms"
        )


if __name__ == "__main__":
    main()
