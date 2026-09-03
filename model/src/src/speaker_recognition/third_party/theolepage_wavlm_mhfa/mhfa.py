"""Multi-head factorized attentive pooling from the pinned upstream source.

See ``SOURCE.md`` and ``LICENSE.md`` in this package for provenance and terms.
The parameter names and mathematical operations remain checkpoint-compatible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class MHFA(nn.Module):
    """Pool all WavLM representation levels into one speaker embedding."""

    def __init__(
        self,
        *,
        head_nb: int = 8,
        inputs_dim: int = 768,
        compression_dim: int = 128,
        outputs_dim: int = 256,
        representation_levels: int = 13,
    ) -> None:
        """Create layer weights, compression projections, and attention heads."""
        super().__init__()
        self.weights_k = nn.Parameter(torch.ones(representation_levels))
        self.weights_v = nn.Parameter(torch.ones(representation_levels))
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)
        self.pooling_fc = nn.Linear(
            self.head_nb * self.cmp_dim,
            self.ous_dim,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pool ``[batch, dimension, frames, levels]`` representations."""
        if inputs.ndim != 4:
            raise ValueError(
                "MHFA input must have shape [batch, dimension, frames, levels]."
            )
        if inputs.shape[1] != self.ins_dim:
            raise ValueError(
                f"MHFA expected feature dimension {self.ins_dim}, received "
                f"{inputs.shape[1]}."
            )
        if inputs.shape[-1] != self.weights_k.numel():
            raise ValueError(
                f"MHFA expected {self.weights_k.numel()} representation levels, "
                f"received {inputs.shape[-1]}."
            )

        key = torch.sum(
            inputs * functional.softmax(self.weights_k, dim=-1),
            dim=-1,
        ).transpose(1, 2)
        value = torch.sum(
            inputs * functional.softmax(self.weights_v, dim=-1),
            dim=-1,
        ).transpose(1, 2)

        key = self.cmp_linear_k(key)
        value = self.cmp_linear_v(value).unsqueeze(-2)
        attention = self.att_head(key)
        pooled = torch.sum(
            value * functional.softmax(attention, dim=1).unsqueeze(-1),
            dim=1,
        )
        return self.pooling_fc(pooled.reshape(pooled.shape[0], -1))
