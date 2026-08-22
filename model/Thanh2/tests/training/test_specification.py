"""Dependency-free tests for shared loss and model update specifications."""

from __future__ import annotations

from copy import deepcopy

import pytest

from speaker_recognition.training.specification import (
    AamSoftmaxSpec,
    OptimizationSpec,
    TrainingSpecification,
    TrainingSpecificationError,
)


def _resolved_values(model_name: str = "ecapa_tdnn") -> dict[str, object]:
    """Return one minimal resolved mapping for policy tests."""
    if model_name == "wavlm_mhfa":
        optimization: dict[str, object] = {
            "name": "adamw",
            "trainable_scope": "transformer_layers_mhfa_and_new_head",
            "transformer_learning_rate": 0.00002,
            "backend_learning_rate": 0.005,
            "head_learning_rate": 0.005,
            "layerwise_learning_rate_decay": 1.0,
            "weight_decay": 0.0,
            "selection_status": "candidate_pending_validation",
        }
    else:
        optimization = {
            "name": "adam",
            "trainable_scope": "complete_encoder_and_new_head",
            "encoder_learning_rate": 0.0001,
            "head_learning_rate": 0.0001,
            "weight_decay": 0.000002,
            "selection_status": "candidate_pending_validation",
        }
    return {
        "model": {"name": model_name},
        "training": {
            "mixed_precision": "fp16",
            "gradient_clip_norm": 5.0,
            "batch": {
                "size": 6 if model_name == "wavlm_mhfa" else 24,
                "calibrated_largest_passing_size": (
                    8 if model_name == "wavlm_mhfa" else 32
                ),
                "selection_status": (
                    "memory_calibrated_pending_multibatch_validation"
                ),
                "calibration_artifact": "fixture.json",
            },
            "objective": {
                "name": "aam_softmax",
                "margin": 0.2,
                "scale": 30.0,
                "easy_margin": False,
            },
        },
        "optimization": optimization,
    }


def test_shared_aam_softmax_spec_is_parsed_exactly() -> None:
    """The invariant comparison loss should preserve its angular constants."""
    specification = TrainingSpecification.from_resolved_config(
        _resolved_values()
    )

    assert specification.objective == AamSoftmaxSpec(
        margin=0.2,
        scale=30.0,
        easy_margin=False,
    )
    assert specification.mixed_precision == "fp16"
    assert specification.gradient_clip_norm == pytest.approx(5.0)
    assert specification.batch.size == 24
    assert specification.batch.calibrated_largest_passing_size == 32


def test_ecapa_and_rawnet_use_complete_encoder_policy() -> None:
    """Waveform/CNN encoders should expose one complete encoder group."""
    for model_name in ("ecapa_tdnn", "rawnet3"):
        specification = TrainingSpecification.from_resolved_config(
            _resolved_values(model_name)
        )
        assert specification.optimization.trainable_scope == (
            "complete_encoder_and_new_head"
        )
        assert specification.optimization.encoder_learning_rate == pytest.approx(
            0.0001
        )


def test_wavlm_policy_retains_source_learning_rate_separation() -> None:
    """Transformer fine-tuning must stay slower than the target backend/head."""
    specification = TrainingSpecification.from_resolved_config(
        _resolved_values("wavlm_mhfa")
    ).optimization

    assert specification.transformer_learning_rate == pytest.approx(0.00002)
    assert specification.backend_learning_rate == pytest.approx(0.005)
    assert specification.head_learning_rate == pytest.approx(0.005)
    assert specification.layerwise_learning_rate_decay == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("margin", 0.0),
        ("margin", 1.7),
        ("scale", -30.0),
        ("easy_margin", "false"),
    ],
)
def test_reject_invalid_objective_values(field: str, value: object) -> None:
    """Invalid angular geometry must fail before a GPU run starts."""
    values = _resolved_values()
    objective = values["training"]["objective"]  # type: ignore[index]
    objective[field] = value  # type: ignore[index]

    with pytest.raises(TrainingSpecificationError):
        TrainingSpecification.from_resolved_config(values)


def test_reject_wavlm_complete_encoder_scope() -> None:
    """The source-preserved frozen WavLM front end must not be enabled silently."""
    values = _resolved_values("wavlm_mhfa")
    optimization = values["optimization"]
    optimization["trainable_scope"] = (  # type: ignore[index]
        "complete_encoder_and_new_head"
    )

    with pytest.raises(TrainingSpecificationError, match="Transformer"):
        TrainingSpecification.from_resolved_config(values)


def test_reject_layerwise_decay_above_one() -> None:
    """A decay must not give early generic layers the largest update rate."""
    values = deepcopy(_resolved_values("wavlm_mhfa"))
    optimization = values["optimization"]
    optimization["layerwise_learning_rate_decay"] = 1.1  # type: ignore[index]

    with pytest.raises(TrainingSpecificationError, match=r"in \(0, 1\]"):
        TrainingSpecification.from_resolved_config(values)


def test_reject_unknown_model_policy() -> None:
    """New architectures require an explicit reviewed optimizer policy."""
    with pytest.raises(TrainingSpecificationError, match="Unsupported"):
        OptimizationSpec.from_mapping(
            "unknown",
            {
                "name": "adam",
                "trainable_scope": "complete_encoder_and_new_head",
                "head_learning_rate": 0.001,
                "weight_decay": 0.0,
                "selection_status": "pending_validation",
            },
        )
