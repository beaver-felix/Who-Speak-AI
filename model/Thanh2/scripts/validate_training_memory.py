"""Validate downloaded T4 training-memory evidence without PyTorch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


EXPECTED_MODELS = {
    "ecapa_tdnn": {"embedding_dim": 192, "parameter_count": 20_767_552},
    "rawnet3": {"embedding_dim": 256, "parameter_count": 16_280_322},
    "wavlm_mhfa": {"embedding_dim": 256, "parameter_count": 96_684_490},
}


def parse_arguments() -> argparse.Namespace:
    """Parse one or more downloaded calibration artifacts."""
    parser = argparse.ArgumentParser(
        description="Validate schema-2 T4 memory-calibration evidence."
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    return parser.parse_args()


def validate_artifact(path: Path) -> dict[str, Any]:
    """Return a compact accepted summary or raise on any invalid measurement."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 2:
        raise ValueError(f"{path.name}: schema_version must equal 2.")
    if payload.get("purpose") != "memory_calibration_not_model_selection":
        raise ValueError(f"{path.name}: invalid calibration purpose.")

    model = _mapping(payload, "model", path)
    model_name = model.get("name")
    expected = EXPECTED_MODELS.get(str(model_name))
    if expected is None:
        raise ValueError(f"{path.name}: unsupported model {model_name!r}.")
    for field, expected_value in expected.items():
        if model.get(field) != expected_value:
            raise ValueError(f"{path.name}: invalid model {field}.")

    objective = _mapping(payload, "objective", path)
    if objective.get("num_classes") != 10_291:
        raise ValueError(f"{path.name}: conservative class count must be 10291.")
    if objective.get("margin") != 0.2 or objective.get("scale") != 30.0:
        raise ValueError(f"{path.name}: shared AAM control changed.")

    measurements = payload.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise ValueError(f"{path.name}: measurements must be non-empty.")
    passing_sizes: list[int] = []
    for measurement in measurements:
        if not isinstance(measurement, Mapping):
            raise ValueError(f"{path.name}: malformed measurement.")
        if measurement.get("status") != "passed":
            continue
        batch_size = measurement.get("batch_size")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError(f"{path.name}: invalid passing batch size.")
        for field in ("loss", "gradient_norm_before_clipping"):
            value = measurement.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{path.name}: {field} must be finite.")
        if measurement.get("loss_finite") is not True:
            raise ValueError(f"{path.name}: loss finite check failed.")
        if measurement.get("gradient_norm_finite") is not True:
            raise ValueError(f"{path.name}: gradient finite check failed.")
        if measurement.get("optimizer_step_applied") is not True:
            raise ValueError(f"{path.name}: optimizer step was not applied.")
        if not _positive_integer(measurement.get("peak_allocated_bytes")):
            raise ValueError(f"{path.name}: invalid allocated-memory peak.")
        if not _positive_integer(measurement.get("peak_reserved_bytes")):
            raise ValueError(f"{path.name}: invalid reserved-memory peak.")
        passing_sizes.append(batch_size)

    if not passing_sizes:
        raise ValueError(f"{path.name}: no valid passing batch size.")
    largest = max(passing_sizes)
    if payload.get("largest_passing_batch_size") != largest:
        raise ValueError(f"{path.name}: largest passing batch is inconsistent.")
    return {
        "model": model_name,
        "largest_passing_batch_size": largest,
        "conservative_candidate_batch_size": max(1, int(largest * 0.8)),
    }


def _mapping(
    values: Mapping[str, Any],
    key: str,
    path: Path,
) -> Mapping[str, Any]:
    """Read one required mapping with an artifact-specific error."""
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name}: {key} must be a mapping.")
    return value


def _positive_integer(value: object) -> bool:
    """Return whether a JSON value is a strict positive integer."""
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def main() -> None:
    """Validate every artifact and print its conservative next candidate."""
    arguments = parse_arguments()
    seen_models: set[str] = set()
    for artifact in arguments.artifacts:
        summary = validate_artifact(artifact.resolve())
        model_name = str(summary["model"])
        if model_name in seen_models:
            raise ValueError(f"Duplicate model evidence: {model_name}.")
        seen_models.add(model_name)
        print(
            f"{model_name}: largest={summary['largest_passing_batch_size']}, "
            "conservative_candidate="
            f"{summary['conservative_candidate_batch_size']}"
        )
    print("TRAINING MEMORY EVIDENCE VALIDATED")


if __name__ == "__main__":
    main()
