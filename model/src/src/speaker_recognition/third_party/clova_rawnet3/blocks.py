"""RawNet3 residual blocks adapted from Clova's MIT-licensed source.

See ``SOURCE.md`` and ``LICENSE.md`` in this package for provenance and terms.
The mathematical operations match the pinned upstream implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional


class PreEmphasis(torch.nn.Module):
    """Apply the upstream first-order pre-emphasis filter."""

    def __init__(self, coefficient: float = 0.97) -> None:
        """Create the fixed, flipped cross-correlation kernel."""
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer(
            "flipped_filter",
            torch.tensor([-coefficient, 1.0], dtype=torch.float32).view(1, 1, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Filter a two-dimensional ``[batch, time]`` waveform tensor."""
        if inputs.ndim != 2:
            raise ValueError("PreEmphasis input must have shape [batch, time].")
        expanded = inputs.unsqueeze(1)
        padded = functional.pad(expanded, (1, 0), "reflect")
        return functional.conv1d(padded, self.flipped_filter)


class AFMS(nn.Module):
    """Apply alpha-feature-map scaling after a residual block."""

    def __init__(self, dimensions: int) -> None:
        """Create the learned scale offset and channel gate."""
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((dimensions, 1)))
        self.fc = nn.Linear(dimensions, dimensions)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Gate each channel using its global temporal average."""
        gate = functional.adaptive_avg_pool1d(inputs, 1).view(inputs.size(0), -1)
        gate = self.sigmoid(self.fc(gate)).view(
            inputs.size(0), inputs.size(1), -1
        )
        return (inputs + self.alpha) * gate


class Bottle2neck(nn.Module):
    """Implement the Res2Net-style RawNet3 residual block."""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int,
        dilation: int,
        scale: int = 4,
        pool: int | bool = False,
    ) -> None:
        """Build the exact convolution, pooling, residual, and AFMS path."""
        super().__init__()
        width = int(math.floor(planes / scale))

        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1

        padding = math.floor(kernel_size / 2) * dilation
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
                for _ in range(self.nums)
            ]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(width) for _ in range(self.nums)]
        )

        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.max_pool = nn.MaxPool1d(pool) if pool else None
        self.afms = AFMS(planes)
        if inplanes != planes:
            self.residual = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the upstream split-and-add residual computation."""
        residual = self.residual(inputs)
        output = self.bn1(self.relu(self.conv1(inputs)))

        splits = torch.split(output, self.width, 1)
        processed: torch.Tensor | None = None
        aggregated: torch.Tensor | None = None
        for index in range(self.nums):
            processed = (
                splits[index]
                if index == 0
                else processed + splits[index]  # type: ignore[operator]
            )
            processed = self.bns[index](
                self.relu(self.convs[index](processed))
            )
            aggregated = (
                processed
                if aggregated is None
                else torch.cat((aggregated, processed), 1)
            )

        if aggregated is None:
            raise RuntimeError("Bottle2neck scale must create at least one split.")
        output = torch.cat((aggregated, splits[self.nums]), 1)
        output = self.bn3(self.relu(self.conv3(output)))
        output = output + residual
        if self.max_pool is not None:
            output = self.max_pool(output)
        return self.afms(output)
