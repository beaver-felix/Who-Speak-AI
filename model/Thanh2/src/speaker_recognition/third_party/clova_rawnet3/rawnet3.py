"""RawNet3 architecture adapted from Clova's MIT-licensed source.

See ``SOURCE.md`` and ``LICENSE.md`` in this package for provenance and terms.
The layer topology and forward computation match the pinned upstream source.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from asteroid_filterbanks import Encoder, ParamSincFB

from speaker_recognition.third_party.clova_rawnet3.blocks import (
    Bottle2neck,
    PreEmphasis,
)


class RawNet3(nn.Module):
    """Official RawNet3 waveform speaker encoder."""

    def __init__(
        self,
        block: type[nn.Module],
        model_scale: int,
        context: bool,
        summed: bool,
        channels: int = 1024,
        **kwargs: object,
    ) -> None:
        """Build the architecture using the official ECA recipe arguments."""
        super().__init__()
        output_dimensions = int(kwargs["nOut"])
        self.context = context
        self.encoder_type = str(kwargs["encoder_type"])
        self.log_sinc = bool(kwargs["log_sinc"])
        self.norm_sinc = str(kwargs["norm_sinc"])
        self.out_bn = bool(kwargs["out_bn"])
        self.summed = summed

        self.preprocess = nn.Sequential(
            PreEmphasis(), nn.InstanceNorm1d(1, eps=1e-4, affine=True)
        )
        self.conv1 = Encoder(
            ParamSincFB(
                channels // 4,
                251,
                stride=int(kwargs["sinc_stride"]),
            )
        )
        self.relu = nn.ReLU()
        # Kept because it is present in the official state dictionary, even
        # though the pinned upstream forward method does not call it.
        self.bn1 = nn.BatchNorm1d(channels // 4)

        self.layer1 = block(
            channels // 4,
            channels,
            kernel_size=3,
            dilation=2,
            scale=model_scale,
            pool=5,
        )
        self.layer2 = block(
            channels,
            channels,
            kernel_size=3,
            dilation=3,
            scale=model_scale,
            pool=3,
        )
        self.layer3 = block(
            channels,
            channels,
            kernel_size=3,
            dilation=4,
            scale=model_scale,
        )
        self.layer4 = nn.Conv1d(3 * channels, 1536, kernel_size=1)

        attention_input = 1536 * 3 if self.context else 1536
        if self.encoder_type == "ECA":
            attention_output = 1536
        elif self.encoder_type == "ASP":
            attention_output = 1
        else:
            raise ValueError(f"Undefined RawNet3 encoder: {self.encoder_type!r}.")

        self.attention = nn.Sequential(
            nn.Conv1d(attention_input, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, attention_output, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, output_dimensions)
        self.bn6 = nn.BatchNorm1d(output_dimensions)
        self.mp3 = nn.MaxPool1d(3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode a ``[batch, samples]`` waveform tensor."""
        # The upstream implementation deliberately runs the Sinc front end in
        # float32 outside mixed precision for numerical stability.
        with torch.amp.autocast("cuda", enabled=False):
            output = self.preprocess(inputs)
            output = torch.abs(self.conv1(output))
            if self.log_sinc:
                output = torch.log(output + 1e-6)
            if self.norm_sinc == "mean":
                output = output - torch.mean(output, dim=-1, keepdim=True)
            elif self.norm_sinc == "mean_std":
                mean = torch.mean(output, dim=-1, keepdim=True)
                standard_deviation = torch.std(output, dim=-1, keepdim=True)
                standard_deviation = standard_deviation.clamp(min=0.001)
                output = (output - mean) / standard_deviation

        first = self.layer1(output)
        second = self.layer2(first)
        third_input = self.mp3(first) + second if self.summed else second
        third = self.layer3(third_input)

        output = self.relu(
            self.layer4(torch.cat((self.mp3(first), second, third), dim=1))
        )
        time_steps = output.size(-1)
        if self.context:
            global_output = torch.cat(
                (
                    output,
                    torch.mean(output, dim=2, keepdim=True).repeat(
                        1, 1, time_steps
                    ),
                    torch.sqrt(
                        torch.var(output, dim=2, keepdim=True).clamp(
                            min=1e-4, max=1e4
                        )
                    ).repeat(1, 1, time_steps),
                ),
                dim=1,
            )
        else:
            global_output = output

        weights = self.attention(global_output)
        mean = torch.sum(output * weights, dim=2)
        standard_deviation = torch.sqrt(
            (
                torch.sum((output**2) * weights, dim=2) - mean**2
            ).clamp(min=1e-4, max=1e4)
        )
        pooled = self.bn5(torch.cat((mean, standard_deviation), 1))
        embedding = self.fc6(pooled)
        return self.bn6(embedding) if self.out_bn else embedding


def build_rawnet3(
    *,
    output_dimensions: int = 256,
    encoder_type: str = "ECA",
    sinc_stride: int = 10,
) -> RawNet3:
    """Instantiate the exact architecture used by the pinned checkpoint."""
    return RawNet3(
        Bottle2neck,
        model_scale=8,
        context=True,
        summed=True,
        out_bn=False,
        log_sinc=True,
        norm_sinc="mean",
        grad_mult=1,
        nOut=output_dimensions,
        encoder_type=encoder_type,
        sinc_stride=sinc_stride,
    )
