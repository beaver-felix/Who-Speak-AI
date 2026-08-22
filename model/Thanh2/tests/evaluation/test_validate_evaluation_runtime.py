"""Tests for the dependency-free real evaluation evidence validator."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts/validate_evaluation_runtime.py"


def _load_validator() -> ModuleType:
    """Load the standalone validator without model runtime dependencies."""
    specification = importlib.util.spec_from_file_location(
        "validate_evaluation_runtime",
        SCRIPT_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _payload(model_name: str) -> dict[str, object]:
    """Create complete finite schema-1 evidence for one supported adapter."""
    dimensions = {
        "ecapa_tdnn": (192, 48_000, 4),
        "rawnet3": (256, 64_240, 4),
        "wavlm_mhfa": (256, 64_240, 2),
    }
    embedding_dim, segment_samples, batch_size = dimensions[model_name]
    metrics = {
        "eer": 0.25,
        "min_dcf": 0.5,
        "far": 0.25,
        "frr": 0.25,
        "tar": 0.75,
        "accuracy": 0.75,
        "tar_at_far_5pct": 0.5,
        "tar_at_far_1pct": 0.25,
        "tar_at_far_0p1pct": 0.0,
        "tar_at_far_0p01pct": 0.0,
    }
    checks = {
        "trial_fingerprint_matches": True,
        "exact_trial_count": True,
        "exact_genuine_count": True,
        "exact_impostor_count": True,
        "exact_utterance_count": True,
        "valid_variable_crop_coverage": True,
        "complete_required_metrics": True,
        "metrics_finite": True,
        "security_threshold_not_selected": True,
        "returned_metrics_finite": True,
        "latency_finite_positive": True,
    }
    trial_hash = "a" * 64
    return {
        "schema_version": 1,
        "purpose": "bounded_real_audio_evaluation_runtime_gate",
        "model": {
            "name": model_name,
            "source_id": "pinned/source",
            "revision": "b" * 40,
            "embedding_dim": embedding_dim,
        },
        "fixture": {
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
            "trial_list_sha256": trial_hash,
        },
        "runtime": {
            "device": "cuda:0",
            "device_name": "Tesla T4",
            "torch_version": "2.10.0+cu128",
            "cuda_version": "12.8",
            "maximum_allocated_cuda_bytes": 1_000_000,
        },
        "validation": {
            "protocol": {
                "trial_list_sha256": trial_hash,
                "trial_count": 16,
            },
            "embeddings": {
                "utterance_count": 8,
                "embedding_table_sha256": "c" * 64,
            },
            "threshold_policy": {"security_threshold_selected": False},
            "metrics": metrics,
            "evaluation": {
                "embedding_extraction": {
                    "utterance_count": 8,
                    "crop_count": 12,
                    "batch_count": 2,
                    "wall_seconds": 1.0,
                    "model_seconds": 0.5,
                }
            },
        },
        "latency": {
            "warmup_iterations": 10,
            "measured_iterations": 50,
            "mean_ms": 5.0,
            "median_ms": 4.8,
            "p95_ms": 6.0,
        },
        "checks": checks,
    }


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    """Write one temporary JSON artifact for validator tests."""
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "model_name",
    ["ecapa_tdnn", "rawnet3", "wavlm_mhfa"],
)
def test_validator_accepts_complete_model_evidence(
    tmp_path: Path,
    model_name: str,
) -> None:
    """Every supported adapter must share the same evidence contract."""
    summary = _load_validator().validate_artifact(
        _write(tmp_path, _payload(model_name))
    )

    assert summary == {
        "model": model_name,
        "eer": 0.25,
        "min_dcf": 0.5,
        "median_latency_ms": 4.8,
    }


def test_validator_rejects_test_threshold_selection(tmp_path: Path) -> None:
    """A smoke gate cannot claim that it selected a security threshold."""
    payload = _payload("ecapa_tdnn")
    payload["validation"]["threshold_policy"][
        "security_threshold_selected"
    ] = True

    with pytest.raises(ValueError, match="cannot select"):
        _load_validator().validate_artifact(_write(tmp_path, payload))


def test_validator_rejects_non_finite_latency(tmp_path: Path) -> None:
    """Timing evidence with NaN must fail closed."""
    payload = _payload("rawnet3")
    payload["latency"]["p95_ms"] = float("nan")

    with pytest.raises(ValueError, match="finite and positive"):
        _load_validator().validate_artifact(_write(tmp_path, payload))


def test_validator_rejects_false_gate_check(tmp_path: Path) -> None:
    """Every producer-side runtime check must remain present and true."""
    payload = _payload("wavlm_mhfa")
    payload["checks"]["valid_variable_crop_coverage"] = False

    with pytest.raises(ValueError, match="checks failed"):
        _load_validator().validate_artifact(_write(tmp_path, payload))


@pytest.mark.parametrize("crop_count", [7, 17])
def test_validator_rejects_crop_count_outside_policy(
    tmp_path: Path,
    crop_count: int,
) -> None:
    """Eight utterances may contribute only one or two crops apiece."""
    payload = _payload("ecapa_tdnn")
    payload["validation"]["evaluation"]["embedding_extraction"][
        "crop_count"
    ] = crop_count

    with pytest.raises(ValueError, match="one or two crops"):
        _load_validator().validate_artifact(_write(tmp_path, payload))
