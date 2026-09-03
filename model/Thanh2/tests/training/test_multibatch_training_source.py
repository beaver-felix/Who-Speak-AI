"""Dependency-free source checks for the real multi-batch Kaggle gate."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "smoke_test_multibatch_training.py"


def test_multibatch_gate_parses_without_model_dependencies() -> None:
    """The committed Kaggle entry point must remain syntactically valid."""
    ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"), filename=str(SCRIPT_PATH))


def test_gate_uses_canonical_real_data_pipeline() -> None:
    """The gate must not fall back to repeated synthetic waveform tensors."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    for required in (
        "parse_tidyvoice_audio_path",
        "TrainingSpeakerDataset",
        "collate_training_samples",
        "torch.utils.data.DataLoader",
        "distinct_speakers",
        "distinct_utterances",
    ):
        assert required in source
    assert ".repeat(batch_size" not in source


def test_gate_requires_finite_applied_updates_for_every_group() -> None:
    """Passing evidence must prove encoder/backend and head updates."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    for required in (
        "gradient_norm_finite",
        "_capture_group_probes",
        "_updated_probe_groups",
        "all_optimizer_groups_updated_across_gate",
        "mandatory_groups_updated_each_step",
        "loss_scale_never_backed_off",
        "allow_nan=False",
    ):
        assert required in source


def test_gate_uses_accepted_candidate_batches_from_config() -> None:
    """The script must consume the reviewed config instead of hardcoded batches."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "batch_size = specification.batch.size" in source
    assert "num_classes=full_speaker_count" in source
    assert "TIDYVOICE_TRAIN_SPEAKERS = 3_666" in source


def test_gate_preserves_official_wavlm_layerdrop_behavior() -> None:
    """Individual Transformer skips are allowed but all must update overall."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "layerdrop 0.05" in source
    assert "groups_without_nonzero_gradient" in source
    assert '{"mhfa", "aam_softmax_head"}' in source


def test_gate_can_prove_fp32_fallback_memory_and_updates() -> None:
    """Correction gate must support explicit FP32 without changing defaults."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert '("fp16", "fp32")' in source
    assert 'default="fp16"' in source
    assert 'enabled=arguments.precision == "fp16"' in source
    assert '"mixed_precision": arguments.precision' in source
