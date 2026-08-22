"""Tests for deterministic layered experiment configuration."""

import json
from pathlib import Path

import pytest

from speaker_recognition.configuration import (
    ConfigurationError,
    resolve_layered_config,
    write_resolved_config,
)


@pytest.mark.parametrize(
    ("dataset_name", "model_name", "embedding_dim"),
    [
        ("tidyvoice", "ecapa_tdnn", 192),
        ("tidyvoice", "rawnet3", 256),
        ("tidyvoice", "wavlm_mhfa", 256),
        ("vimd", "ecapa_tdnn", 192),
        ("vimd", "rawnet3", 256),
        ("vimd", "wavlm_mhfa", 256),
    ],
)
def test_repository_layers_resolve_all_six_experiments(
    dataset_name: str,
    model_name: str,
    embedding_dim: int,
) -> None:
    """Every required dataset/model combination should resolve completely."""
    root = Path(__file__).parents[1]
    resolved = resolve_layered_config(
        (
            root / "configs/base.toml",
            root / f"configs/datasets/{dataset_name}.toml",
            root / f"configs/models/{model_name}.toml",
        )
    )

    assert resolved.get("audio.sample_rate") == 16000
    assert resolved.get("data.name") == dataset_name
    assert resolved.get("model.name") == model_name
    assert resolved.get("model.embedding_dim") == embedding_dim
    assert resolved.get("training.objective.margin") == pytest.approx(0.2)
    assert resolved.get("training.objective.scale") == pytest.approx(30.0)
    assert resolved.get("optimization.selection_status").endswith(
        "pending_validation"
    )
    expected_batch_size = 6 if model_name == "wavlm_mhfa" else 24
    assert resolved.get("training.batch.size") == expected_batch_size
    assert resolved.get("training.batch.selection_status") == (
        "memory_calibrated_pending_multibatch_validation"
    )
    assert len(resolved.source_paths) == 3
    assert len(resolved.sha256) == 64


def test_later_layers_and_explicit_overrides_replace_existing_leaves(
    tmp_path: Path,
) -> None:
    """Merge precedence should be deterministic and preserve value types."""
    base, dataset, model = _write_minimal_layers(tmp_path)
    resolved = resolve_layered_config(
        (base, dataset, model),
        overrides={"experiment.stage": "optimized"},
    )

    assert resolved.get("experiment.stage") == "optimized"
    assert resolved.get("training.learning_rate") == pytest.approx(0.0002)


def test_fingerprint_is_independent_of_override_mapping_order(
    tmp_path: Path,
) -> None:
    """Equivalent typed values must produce the same canonical fingerprint."""
    layers = _write_minimal_layers(tmp_path)
    first = resolve_layered_config(
        layers,
        overrides={
            "experiment.stage": "screening",
            "training.learning_rate": 0.0003,
        },
    )
    second = resolve_layered_config(
        layers,
        overrides={
            "training.learning_rate": 0.0003,
            "experiment.stage": "screening",
        },
    )

    assert first.sha256 == second.sha256
    assert first.to_dict() == second.to_dict()


def test_reject_unknown_override_path(tmp_path: Path) -> None:
    """A misspelled override must fail instead of becoming dead config."""
    layers = _write_minimal_layers(tmp_path)

    with pytest.raises(ConfigurationError, match="Unknown"):
        resolve_layered_config(
            layers,
            overrides={"training.learnnig_rate": 0.5},
        )


def test_reject_table_scalar_structure_conflict(tmp_path: Path) -> None:
    """A layer must not replace a complete table with one scalar."""
    layers = list(_write_minimal_layers(tmp_path))
    conflicting = tmp_path / "conflicting.toml"
    conflicting.write_text('training = "invalid"\n', encoding="utf-8")

    with pytest.raises(ConfigurationError, match="structure conflict"):
        resolve_layered_config((*layers, conflicting))


def test_reject_change_to_accepted_trial_protocol(tmp_path: Path) -> None:
    """Resolved runs must retain the fingerprinted shared trial settings."""
    layers = _write_minimal_layers(tmp_path)

    with pytest.raises(ConfigurationError, match="Decision 005"):
        resolve_layered_config(
            layers,
            overrides={"verification.impostor_trials_per_split": 9999},
        )


def test_write_resolved_config_includes_hash_and_sources(
    tmp_path: Path,
) -> None:
    """Saved run metadata should contain values and their provenance."""
    resolved = resolve_layered_config(_write_minimal_layers(tmp_path))
    output = tmp_path / "resolved.json"

    write_resolved_config(resolved, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["config_sha256"] == resolved.sha256
    assert payload["config"] == resolved.to_dict()
    assert len(payload["source_layers"]) == 3


def _write_minimal_layers(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a complete three-layer fixture."""
    base = tmp_path / "base.toml"
    dataset = tmp_path / "dataset.toml"
    model = tmp_path / "model.toml"
    base.write_text(
        """
schema_version = 1
[experiment]
seed = 42
stage = "smoke"
[audio]
sample_rate = 16000
amplitude_normalization = "none"
channel_policy = "arithmetic_mean"
resampler = "scipy_resample_poly"
[loader]
persistent_workers = false
[training]
mixed_precision = "fp16"
gradient_clip_norm = 5.0
learning_rate = 0.001
[training.objective]
name = "aam_softmax"
margin = 0.2
scale = 30.0
easy_margin = false
selection_status = "shared_control_accepted_pending_margin_ablation"
[verification]
seed = 42
max_genuine_per_speaker = 20
impostor_trials_per_split = 100000
[metrics]
far_targets = [0.05, 0.01, 0.001, 0.0001]
[metrics.min_dcf]
p_target = 0.01
c_miss = 1.0
c_false_alarm = 1.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    dataset.write_text(
        """
[data]
name = "fixture"
source_id = "fixture/dataset"
storage = "file"
split_protocol = "fixture.json"
[training]
learning_rate = 0.0002
""".strip()
        + "\n",
        encoding="utf-8",
    )
    model.write_text(
        """
[model]
name = "ecapa_tdnn"
source_id = "fixture/model"
revision = "0123456789abcdef"
embedding_dim = 64
[optimization]
name = "adam"
trainable_scope = "complete_encoder_and_new_head"
encoder_learning_rate = 0.0001
head_learning_rate = 0.0001
weight_decay = 0.000002
selection_status = "fixture_pending_validation"
[training.batch]
size = 8
calibrated_largest_passing_size = 10
selection_status = "memory_calibrated_pending_multibatch_validation"
calibration_artifact = "fixture.json"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return base, dataset, model
