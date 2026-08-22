"""Explicit optimizer groups and freezing policy for the three adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - Kaggle dependency gate.
    raise ModuleNotFoundError(
        "Optimizer construction requires Kaggle's CUDA-matched PyTorch build."
    ) from error

from speaker_recognition.training.specification import OptimizationSpec


@dataclass(frozen=True, slots=True)
class OptimizerBundle:
    """Return the optimizer together with auditable parameter-group metadata."""

    optimizer: torch.optim.Optimizer
    group_names: tuple[str, ...]
    group_parameter_counts: tuple[int, ...]


def build_optimizer(
    adapter: torch.nn.Module,
    objective: torch.nn.Module,
    specification: OptimizationSpec,
) -> OptimizerBundle:
    """Freeze the intended scope and construct non-overlapping update groups."""
    metadata = getattr(adapter, "metadata", None)
    model_name = getattr(metadata, "name", None)
    _set_all_trainable(adapter, False)
    _set_all_trainable(objective, True)

    groups: list[dict[str, Any]] = []
    names: list[str] = []
    if model_name in {"ecapa_tdnn", "rawnet3"}:
        _set_all_trainable(adapter, True)
        _append_group(
            groups,
            names,
            name="encoder",
            parameters=adapter.parameters(),
            learning_rate=_required_rate(
                specification.encoder_learning_rate,
                "encoder_learning_rate",
            ),
        )
    elif model_name == "wavlm_mhfa":
        wavlm = getattr(adapter, "wavlm", None)
        mhfa = getattr(adapter, "mhfa", None)
        layers = getattr(getattr(wavlm, "encoder", None), "layers", None)
        if layers is None or mhfa is None:
            raise ValueError(
                "wavlm_mhfa adapter must expose wavlm.encoder.layers and mhfa."
            )
        transformer_rate = _required_rate(
            specification.transformer_learning_rate,
            "transformer_learning_rate",
        )
        layer_count = len(layers)
        if layer_count == 0:
            raise ValueError("WavLM must contain at least one Transformer layer.")
        for index, layer in enumerate(layers):
            _set_all_trainable(layer, True)
            # Later layers are closer to the speaker objective. For a decay
            # below one they receive the larger rate; decay=1 matches the
            # pinned upstream base recipe exactly.
            exponent = layer_count - index - 1
            _append_group(
                groups,
                names,
                name=f"transformer_layer_{index:02d}",
                parameters=layer.parameters(),
                learning_rate=(
                    transformer_rate
                    * specification.layerwise_learning_rate_decay**exponent
                ),
            )
        _set_all_trainable(mhfa, True)
        _append_group(
            groups,
            names,
            name="mhfa",
            parameters=mhfa.parameters(),
            learning_rate=_required_rate(
                specification.backend_learning_rate,
                "backend_learning_rate",
            ),
        )
    else:
        raise ValueError(f"Unsupported adapter metadata name: {model_name!r}.")

    _append_group(
        groups,
        names,
        name="aam_softmax_head",
        parameters=objective.parameters(),
        learning_rate=specification.head_learning_rate,
    )
    _validate_parameter_partition(adapter, objective, groups)

    optimizer_class = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }[specification.optimizer]
    optimizer = optimizer_class(
        groups,
        weight_decay=specification.weight_decay,
    )
    counts = tuple(
        sum(int(parameter.numel()) for parameter in group["params"])
        for group in groups
    )
    return OptimizerBundle(
        optimizer=optimizer,
        group_names=tuple(names),
        group_parameter_counts=counts,
    )


def _append_group(
    groups: list[dict[str, Any]],
    names: list[str],
    *,
    name: str,
    parameters: Iterable[torch.nn.Parameter],
    learning_rate: float,
) -> None:
    """Materialize one non-empty group so generators cannot be consumed twice."""
    materialized = [parameter for parameter in parameters if parameter.requires_grad]
    if not materialized:
        raise ValueError(f"Optimizer group {name!r} has no trainable parameters.")
    groups.append({"params": materialized, "lr": learning_rate})
    names.append(name)


def _set_all_trainable(module: torch.nn.Module, trainable: bool) -> None:
    """Set one complete module's gradient policy explicitly."""
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def _required_rate(value: float | None, field_name: str) -> float:
    """Narrow an architecture-dependent optional rate after validation."""
    if value is None or value <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return value


def _validate_parameter_partition(
    adapter: torch.nn.Module,
    objective: torch.nn.Module,
    groups: list[dict[str, Any]],
) -> None:
    """Prove that each enabled parameter appears in exactly one group."""
    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise ValueError("A parameter appears in more than one optimizer group.")
    expected_ids = {
        id(parameter)
        for parameter in (*adapter.parameters(), *objective.parameters())
        if parameter.requires_grad
    }
    if set(grouped_ids) != expected_ids:
        raise ValueError(
            "Optimizer groups do not exactly cover all enabled parameters."
        )
