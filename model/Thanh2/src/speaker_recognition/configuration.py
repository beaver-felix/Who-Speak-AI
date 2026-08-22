"""Deterministic layered experiment configuration without extra dependencies."""

from __future__ import annotations

import hashlib
import json
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class ConfigurationError(ValueError):
    """Raised when configuration layers cannot form a valid experiment."""


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """Store an immutable canonical representation of a resolved experiment.

    The canonical JSON string prevents callers from mutating the configuration
    after its fingerprint has been calculated. ``to_dict`` returns a fresh
    copy suitable for W&B logging or runtime component construction.
    """

    _canonical_json: str
    source_paths: tuple[str, ...]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a detached dictionary copy of the resolved values."""
        payload = json.loads(self._canonical_json)
        if not isinstance(payload, dict):  # Defensive: constructor is private.
            raise ConfigurationError("Resolved configuration root is invalid.")
        return payload

    def get(self, dotted_path: str) -> Any:
        """Read one value by a dotted path such as ``audio.sample_rate``."""
        return _read_dotted_path(self.to_dict(), dotted_path)


def resolve_layered_config(
    layer_paths: Sequence[str | Path],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> ResolvedConfig:
    """Load, merge, validate, and fingerprint ordered TOML layers.

    Later files override earlier files. Tables merge recursively, while scalar
    leaves replace scalar leaves. Replacing a table with a scalar, or the
    reverse, is rejected because that usually indicates a misspelled or
    structurally incompatible setting.

    Explicit overrides use existing dotted paths only. This catches command-
    line typos instead of silently introducing unused hyperparameters.
    """
    if not layer_paths:
        raise ConfigurationError("At least one configuration layer is required.")

    resolved: dict[str, Any] = {}
    normalized_paths: list[str] = []
    for raw_path in layer_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ConfigurationError(
                f"Configuration layer does not exist: {path}"
            )
        try:
            with path.open("rb") as stream:
                layer = tomllib.load(stream)
        except tomllib.TOMLDecodeError as error:
            raise ConfigurationError(
                f"Invalid TOML configuration layer {path}: {error}"
            ) from error

        _deep_merge(resolved, layer, path=())
        normalized_paths.append(path.as_posix())

    for dotted_path, value in sorted((overrides or {}).items()):
        _set_existing_dotted_path(resolved, dotted_path, deepcopy(value))

    validate_experiment_config(resolved)
    canonical_json = json.dumps(
        resolved,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    fingerprint = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return ResolvedConfig(
        _canonical_json=canonical_json,
        source_paths=tuple(normalized_paths),
        sha256=fingerprint,
    )


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    """Validate shared identity and comparison-critical configuration fields."""
    if _read_dotted_path(config, "schema_version") != 1:
        raise ConfigurationError("schema_version must equal 1.")

    _require_non_negative_integer(config, "experiment.seed")
    stage = _require_non_empty_text(config, "experiment.stage")
    if stage not in {"unselected", "pilot", "full"}:
        raise ConfigurationError(
            "experiment.stage must be unselected, pilot, or full."
        )

    sample_rate = _require_positive_integer(config, "audio.sample_rate")
    if sample_rate != 16000:
        raise ConfigurationError(
            "audio.sample_rate must equal the accepted canonical 16000 Hz."
        )
    if _read_dotted_path(config, "audio.amplitude_normalization") != "none":
        raise ConfigurationError(
            "audio.amplitude_normalization must remain 'none'."
        )
    if _read_dotted_path(config, "audio.channel_policy") != "arithmetic_mean":
        raise ConfigurationError(
            "audio.channel_policy must remain 'arithmetic_mean'."
        )
    if _read_dotted_path(config, "audio.resampler") != "scipy_resample_poly":
        raise ConfigurationError(
            "audio.resampler must remain 'scipy_resample_poly'."
        )

    _require_non_empty_text(config, "data.name")
    _require_non_empty_text(config, "data.source_id")
    _require_non_empty_text(config, "data.split_protocol")
    storage = _read_dotted_path(config, "data.storage")
    if storage not in {"file", "parquet"}:
        raise ConfigurationError("data.storage must be 'file' or 'parquet'.")

    _require_non_empty_text(config, "model.name")
    _require_non_empty_text(config, "model.source_id")
    _require_non_empty_text(config, "model.revision")
    _require_positive_integer(config, "model.embedding_dim")

    verification_seed = _require_non_negative_integer(
        config,
        "verification.seed",
    )
    genuine_cap = _require_positive_integer(
        config,
        "verification.max_genuine_per_speaker",
    )
    impostor_count = _require_positive_integer(
        config,
        "verification.impostor_trials_per_split",
    )
    if (verification_seed, genuine_cap, impostor_count) != (42, 20, 100_000):
        raise ConfigurationError(
            "Verification settings must match accepted Decision 005: seed 42, "
            "genuine cap 20, and 100000 impostors per split."
        )

    far_targets = _read_dotted_path(config, "metrics.far_targets")
    if far_targets != [0.05, 0.01, 0.001, 0.0001]:
        raise ConfigurationError(
            "metrics.far_targets must match accepted Decision 004."
        )
    expected_min_dcf = {
        "p_target": 0.01,
        "c_miss": 1.0,
        "c_false_alarm": 1.0,
    }
    min_dcf = _read_dotted_path(config, "metrics.min_dcf")
    if min_dcf != expected_min_dcf:
        raise ConfigurationError(
            "metrics.min_dcf must match accepted Decision 004."
        )

    if _read_dotted_path(config, "loader.persistent_workers") is not False:
        raise ConfigurationError(
            "loader.persistent_workers must remain false until epoch state is "
            "synchronized with persistent worker copies."
        )
    group_by_audio_path = _read_dotted_path(
        config,
        "loader.group_by_audio_path",
    )
    if not isinstance(group_by_audio_path, bool):
        raise ConfigurationError(
            "loader.group_by_audio_path must be boolean."
        )
    expected_grouping = storage == "parquet"
    if group_by_audio_path is not expected_grouping:
        raise ConfigurationError(
            "Parquet data requires shard-aware ordering; file data must use "
            "the ordinary epoch shuffle."
        )

    expected_reproducibility = {
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
    }
    if _read_dotted_path(config, "reproducibility") != (
        expected_reproducibility
    ):
        raise ConfigurationError(
            "reproducibility must match the accepted deterministic T4 policy."
        )

    # Decision 010 fixes the comparison objective. Architecture-specific
    # optimizer groups remain explicit model-layer policy and are validated by
    # the dependency-free TrainingSpecification boundary.
    if _read_dotted_path(config, "training.mixed_precision") != "fp16":
        raise ConfigurationError(
            "training.mixed_precision must remain 'fp16' for the T4 baseline."
        )
    gradient_clip_norm = _read_dotted_path(
        config,
        "training.gradient_clip_norm",
    )
    if (
        isinstance(gradient_clip_norm, bool)
        or not isinstance(gradient_clip_norm, (int, float))
        or gradient_clip_norm <= 0
    ):
        raise ConfigurationError(
            "training.gradient_clip_norm must be positive."
        )
    expected_objective = {
        "name": "aam_softmax",
        "margin": 0.2,
        "scale": 30.0,
        "easy_margin": False,
        "selection_status": (
            "shared_control_accepted_pending_margin_ablation"
        ),
    }
    if _read_dotted_path(config, "training.objective") != expected_objective:
        raise ConfigurationError(
            "training.objective must match the shared Decision 010 AAM-Softmax "
            "control."
        )

    # Importing this module remains dependency-free; no PyTorch import occurs.
    from speaker_recognition.training.specification import (
        TrainingSpecification,
        TrainingSpecificationError,
    )

    try:
        specification = TrainingSpecification.from_resolved_config(config)
    except TrainingSpecificationError as error:
        raise ConfigurationError(f"Invalid training specification: {error}") from error

    schedule = _read_dotted_path(config, "training.schedule")
    if schedule != {
        "name": "constant",
        "selection_status": "pilot_control_pending_validation",
    }:
        raise ConfigurationError(
            "The initial controlled experiment requires a constant schedule."
        )
    tracking = _read_dotted_path(config, "tracking")
    if not isinstance(tracking, Mapping):
        raise ConfigurationError("tracking must be a table.")
    _require_non_empty_text(config, "tracking.project")
    tracking_mode = _read_dotted_path(config, "tracking.mode")
    if tracking_mode not in {"online", "offline"}:
        raise ConfigurationError("tracking.mode must be online or offline.")

    if stage == "pilot":
        if (
            specification.epoch_sampling.max_utterances_per_speaker != 1
            or specification.epoch_sampling.max_speakers_per_epoch != 512
            or specification.lifecycle.max_epochs != 1
            or specification.lifecycle.early_stopping_patience != 1
            or specification.evaluation.segment_count != 1
            or tracking_mode != "offline"
        ):
            raise ConfigurationError(
                "Pilot stage must use 512 speakers, one utterance each, one "
                "epoch, patience one, one Validation crop, and offline "
                "tracking."
            )
    elif stage == "full":
        if (
            specification.epoch_sampling.max_utterances_per_speaker != 4
            or specification.epoch_sampling.max_speakers_per_epoch is not None
            or specification.lifecycle.max_epochs != 15
            or specification.lifecycle.early_stopping_patience != 3
            or specification.evaluation.segment_count != 2
            or tracking_mode != "online"
        ):
            raise ConfigurationError(
                "Full stage must use the declared 4-utterance cap, 15 epochs, "
                "patience three, two Validation crops, and online tracking."
            )


def write_resolved_config(
    resolved: ResolvedConfig,
    output_path: str | Path,
) -> None:
    """Write configuration, provenance, and fingerprint as stable JSON."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": resolved.to_dict(),
        "config_sha256": resolved.sha256,
        "source_layers": list(resolved.source_paths),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_resolved_config(path: str | Path) -> ResolvedConfig:
    """Load and authenticate one previously fingerprinted JSON configuration."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError(
            f"Resolved configuration does not exist: {source}"
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError("Resolved configuration JSON is invalid.") from error
    if not isinstance(payload, Mapping):
        raise ConfigurationError(
            "Resolved configuration root must be an object."
        )
    config = payload.get("config")
    source_layers = payload.get("source_layers")
    expected_sha256 = payload.get("config_sha256")
    if not isinstance(config, Mapping):
        raise ConfigurationError("Resolved configuration is missing config.")
    if (
        not isinstance(source_layers, list)
        or not source_layers
        or any(
            not isinstance(layer, str) or not layer.strip()
            for layer in source_layers
        )
    ):
        raise ConfigurationError(
            "Resolved configuration requires non-empty source layers."
        )
    canonical_json = json.dumps(
        config,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    actual_sha256 = hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()
    if expected_sha256 != actual_sha256:
        raise ConfigurationError(
            "Resolved configuration SHA-256 does not match its contents."
        )
    validate_experiment_config(config)
    return ResolvedConfig(
        _canonical_json=canonical_json,
        source_paths=tuple(source_layers),
        sha256=actual_sha256,
    )


def _deep_merge(
    destination: dict[str, Any],
    source: Mapping[str, Any],
    *,
    path: tuple[str, ...],
) -> None:
    """Recursively merge one parsed TOML layer with structural checks."""
    for key, source_value in source.items():
        current_path = (*path, key)
        if key not in destination:
            destination[key] = deepcopy(source_value)
            continue

        destination_value = destination[key]
        source_is_table = isinstance(source_value, Mapping)
        destination_is_table = isinstance(destination_value, dict)
        if source_is_table and destination_is_table:
            _deep_merge(
                destination_value,
                source_value,
                path=current_path,
            )
        elif source_is_table != destination_is_table:
            raise ConfigurationError(
                "Configuration structure conflict at "
                f"{'.'.join(current_path)!r}."
            )
        else:
            destination[key] = deepcopy(source_value)


def _set_existing_dotted_path(
    config: dict[str, Any],
    dotted_path: str,
    value: Any,
) -> None:
    """Replace an existing scalar leaf selected by dotted path."""
    parts = _validate_dotted_path(dotted_path)
    current: dict[str, Any] = config
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            raise ConfigurationError(
                f"Unknown configuration override: {dotted_path!r}."
            )
        current = next_value

    leaf = parts[-1]
    if leaf not in current or isinstance(current[leaf], dict):
        raise ConfigurationError(
            f"Unknown scalar configuration override: {dotted_path!r}."
        )
    if isinstance(value, Mapping):
        raise ConfigurationError(
            f"Override {dotted_path!r} must be a scalar or list value."
        )
    current[leaf] = value


def _read_dotted_path(config: Mapping[str, Any], dotted_path: str) -> Any:
    """Read a required configuration value or raise a bounded error."""
    parts = _validate_dotted_path(dotted_path)
    current: Any = config
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigurationError(
                f"Missing required configuration value: {dotted_path!r}."
            )
        current = current[part]
    return current


def _validate_dotted_path(dotted_path: str) -> tuple[str, ...]:
    """Normalize a non-empty dotted configuration path."""
    if not isinstance(dotted_path, str):
        raise ConfigurationError("Configuration path must be a string.")
    parts = tuple(part.strip() for part in dotted_path.split("."))
    if not parts or any(not part for part in parts):
        raise ConfigurationError(
            f"Invalid dotted configuration path: {dotted_path!r}."
        )
    return parts


def _require_non_empty_text(
    config: Mapping[str, Any],
    dotted_path: str,
) -> str:
    """Require one non-empty string configuration value."""
    value = _read_dotted_path(config, dotted_path)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{dotted_path} must be non-empty text.")
    return value


def _require_non_negative_integer(
    config: Mapping[str, Any],
    dotted_path: str,
) -> int:
    """Require one integer value greater than or equal to zero."""
    value = _read_dotted_path(config, dotted_path)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError(
            f"{dotted_path} must be a non-negative integer."
        )
    return value


def _require_positive_integer(
    config: Mapping[str, Any],
    dotted_path: str,
) -> int:
    """Require one integer value greater than zero."""
    value = _read_dotted_path(config, dotted_path)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{dotted_path} must be a positive integer.")
    return value
