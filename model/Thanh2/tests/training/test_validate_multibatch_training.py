"""Tests for strict real multi-batch evidence validation."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validate_multibatch_training.py"


def _load_validator() -> ModuleType:
    """Load the standalone validator as a test module."""
    specification = importlib.util.spec_from_file_location(
        "validate_multibatch_training",
        SCRIPT_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _payload() -> dict[str, object]:
    """Return one minimal valid ECAPA multi-batch artifact."""
    groups = ["encoder", "aam_softmax_head"]
    step = {
        "batch_size": 24,
        "distinct_speakers": 24,
        "distinct_utterances": 24,
        "loss": 10.0,
        "accuracy": 0.0,
        "gradient_norm_before_clipping": 20.0,
        "audio_finite": True,
        "embeddings_finite": True,
        "scale_before": 1024.0,
        "scale_after": 1024.0,
        "updated_groups": groups,
        "groups_without_nonzero_gradient": [],
        "peak_allocated_bytes": 1_000_000,
        "peak_reserved_bytes": 2_000_000,
        "elapsed_seconds": 1.0,
    }
    steps = [{**step, "step": index} for index in range(1, 4)]
    return {
        "schema_version": 1,
        "purpose": "multirecord_multibatch_training_gate_not_performance",
        "model": {
            "name": "ecapa_tdnn",
            "embedding_dim": 192,
            "parameter_count": 20_767_552,
        },
        "data": {
            "name": "tidyvoice",
            "full_training_speaker_count": 3_666,
            "selected_speaker_count": 72,
            "selected_utterance_count": 72,
            "selected_utterance_ids_sha256": "a" * 64,
            "one_utterance_per_speaker": True,
            "segment_samples": 48_000,
        },
        "training": {
            "batch_size": 24,
            "steps": 3,
            "mixed_precision": "fp16",
            "optimizer": {"group_names": groups},
        },
        "runtime": {
            "device_name": "Tesla T4",
            "torch_version": "2.10.0+cu128",
            "cuda_version": "12.8",
        },
        "steps": steps,
        "checks": {
            "step_count_matches": True,
            "all_speakers_are_distinct": True,
            "all_utterances_are_distinct": True,
            "all_losses_finite": True,
            "all_audio_finite": True,
            "all_embeddings_finite": True,
            "all_gradient_norms_finite": True,
            "all_available_groups_updated": True,
            "mandatory_groups_updated_each_step": True,
            "all_optimizer_groups_updated_across_gate": True,
            "loss_scale_never_backed_off": True,
        },
        "config_sha256": "b" * 64,
    }


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    """Write one evidence fixture."""
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_accept_complete_three_step_evidence(tmp_path: Path) -> None:
    """A complete ECAPA gate should return its bounded summary."""
    summary = _load_validator().validate_artifact(
        _write(tmp_path, _payload())
    )

    assert summary == {
        "model": "ecapa_tdnn",
        "batch_size": 24,
        "steps": 3,
        "distinct_speakers": 72,
    }


def test_reject_repeated_speaker_check(tmp_path: Path) -> None:
    """Evidence cannot pass when identities repeat across selected batches."""
    payload = deepcopy(_payload())
    payload["checks"]["all_speakers_are_distinct"] = False  # type: ignore[index]

    with pytest.raises(ValueError, match="checks failed"):
        _load_validator().validate_artifact(_write(tmp_path, payload))


def test_reject_incomplete_optimizer_group_update(tmp_path: Path) -> None:
    """A classifier-only update must not pass the integration gate."""
    payload = deepcopy(_payload())
    payload["steps"][0]["updated_groups"] = [  # type: ignore[index]
        "aam_softmax_head"
    ]
    payload["steps"][0]["groups_without_nonzero_gradient"] = [  # type: ignore[index]
        "encoder"
    ]

    with pytest.raises(ValueError, match="mandatory group"):
        _load_validator().validate_artifact(_write(tmp_path, payload))


def test_reject_nonfinite_step_value(tmp_path: Path) -> None:
    """NaN and infinity must fail even if the aggregate check says true."""
    payload = deepcopy(_payload())
    payload["steps"][1]["loss"] = float("inf")  # type: ignore[index]

    with pytest.raises(ValueError, match="non-finite loss"):
        _load_validator().validate_artifact(_write(tmp_path, payload))
