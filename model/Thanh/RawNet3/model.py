"""Compact native RawNet3-style speaker encoder.

This implementation is intentionally checkpoint-independent. It follows the
RawNet family design principles—raw waveform input, strided residual temporal
blocks, channel attention, recurrent temporal aggregation, and a compact
speaker embedding—but it is not binary-compatible with an external RawNet3
checkpoint. That makes it safe to train from scratch on a project-specific
speaker manifest and easy to replace with an official checkpoint later.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as torch_functional


class PreEmphasis(nn.Module):
    """Apply a fixed first-order high-pass filter to raw speech."""

    def __init__(self, coefficient: float = 0.97) -> None:
        """Create a non-trainable pre-emphasis filter."""
        super().__init__()
        self.coefficient = coefficient

    def forward(self, waveform: Tensor) -> Tensor:
        """Return a pre-emphasized waveform with the original length."""
        if waveform.ndim != 3:
            raise ValueError("waveform must have shape [batch, channels, time]")
        padded = torch_functional.pad(waveform, (1, 0), mode="reflect")
        return padded[..., 1:] - self.coefficient * padded[..., :-1]


class SqueezeExcitation1D(nn.Module):
    """Learn channel-wise attention from a temporal feature map."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        """Initialize the two-layer channel-gating network."""
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Scale each channel using its global temporal summary."""
        summary = features.mean(dim=-1)
        scale = self.gate(summary).unsqueeze(-1)
        return features * scale


class ResidualTemporalBlock(nn.Module):
    """A strided residual convolutional block for raw-speech features."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        """Build a two-convolution residual block with channel attention."""
        super().__init__()
        padding = dilation
        self.main = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(output_channels),
        )
        self.attention = SqueezeExcitation1D(output_channels)
        if input_channels != output_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(output_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, features: Tensor) -> Tensor:
        """Apply the residual transformation and preserve an identity path."""
        return self.activation(self.attention(self.main(features)) + self.skip(features))


class AdditiveMarginSoftmax(nn.Module):
    """Cosine classifier with an additive margin for speaker classes."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.20,
    ) -> None:
        """Initialize normalized class weights and AM-Softmax parameters."""
        super().__init__()
        if num_classes < 2:
            raise ValueError("at least two speaker classes are required")
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        """Return scaled cosine logits, subtracting the margin for target labels."""
        normalized_embeddings = torch_functional.normalize(embeddings, p=2, dim=-1)
        normalized_weights = torch_functional.normalize(self.weight, p=2, dim=-1)
        cosine = torch.matmul(normalized_embeddings, normalized_weights.transpose(0, 1))
        if labels is not None:
            if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
                raise ValueError("labels must have shape [batch]")
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.long().unsqueeze(1), 1.0)
            cosine = cosine - one_hot * self.margin
        return cosine * self.scale


class RawNet3(nn.Module):
    """Raw waveform speaker encoder with an optional AM-Softmax head."""

    def __init__(
        self,
        num_speakers: Optional[int] = None,
        embedding_dim: int = 192,
        base_channels: int = 64,
        am_scale: float = 30.0,
        am_margin: float = 0.20,
    ) -> None:
        """Create the encoder and optionally a classifier for supervised training."""
        super().__init__()
        if embedding_dim <= 0 or base_channels <= 0:
            raise ValueError("embedding_dim and base_channels must be positive")

        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.pre_emphasis = PreEmphasis()
        self.frontend = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, stride=3, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.residual_stack = nn.Sequential(
            ResidualTemporalBlock(base_channels, base_channels, stride=1),
            ResidualTemporalBlock(base_channels, base_channels * 2, stride=2),
            ResidualTemporalBlock(base_channels * 2, base_channels * 2, stride=1),
            ResidualTemporalBlock(base_channels * 2, base_channels * 4, stride=2),
            ResidualTemporalBlock(base_channels * 4, base_channels * 4, stride=1, dilation=2),
        )
        recurrent_channels = base_channels * 4
        self.temporal_encoder = nn.GRU(
            input_size=recurrent_channels,
            hidden_size=recurrent_channels // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.embedding_layer = nn.Sequential(
            nn.Linear(recurrent_channels * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.classifier = (
            AdditiveMarginSoftmax(embedding_dim, num_speakers, am_scale, am_margin)
            if num_speakers is not None
            else None
        )

    def encode(self, waveform: Tensor) -> Tensor:
        """Encode waveform batches into unnormalized speaker embeddings."""
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("waveform must have shape [batch, time] or [batch, 1, time]")
        waveform = waveform.float()
        features = self.pre_emphasis(waveform)
        features = self.frontend(features)
        features = self.residual_stack(features)
        sequence, _ = self.temporal_encoder(features.transpose(1, 2))
        statistics = torch.cat((sequence.mean(dim=1), sequence.std(dim=1, unbiased=False)), dim=1)
        return self.embedding_layer(statistics)

    def forward(self, waveform: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        """Return embeddings or AM-Softmax logits when labels are supplied."""
        embeddings = self.encode(waveform)
        if labels is not None:
            if self.classifier is None:
                raise RuntimeError("classification head was not created")
            return self.classifier(embeddings, labels)
        return embeddings

    def classification_logits(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """Compute AM-Softmax logits from precomputed embeddings."""
        if self.classifier is None:
            raise RuntimeError("classification head was not created")
        return self.classifier(embeddings, labels)


def extract_speaker_embedding(model: nn.Module, waveform: Tensor) -> Tensor:
    """Return an L2-normalized cosine-ready speaker embedding.

    Args:
        model: A ``RawNet3``-compatible module whose forward pass returns
            embeddings.
        waveform: Audio tensor shaped ``[time]``, ``[batch, time]``, or
            ``[batch, 1, time]``.

    Returns:
        A tensor shaped ``[embedding_dim]`` for one waveform or
        ``[batch, embedding_dim]`` for a batch.
    """
    was_single = waveform.ndim == 1
    if was_single:
        waveform = waveform.unsqueeze(0)
    model_device = next(model.parameters()).device
    waveform = waveform.to(model_device)
    with torch.no_grad():
        embeddings = model(waveform)
    embeddings = torch_functional.normalize(embeddings.float(), p=2, dim=-1)
    return embeddings[0] if was_single else embeddings


def build_model(
    num_speakers: Optional[int],
    embedding_dim: int = 192,
    base_channels: int = 64,
    am_scale: float = 30.0,
    am_margin: float = 0.20,
) -> RawNet3:
    """Construct a RawNet3 encoder from explicit, checkpointable arguments."""
    return RawNet3(
        num_speakers=num_speakers,
        embedding_dim=embedding_dim,
        base_channels=base_channels,
        am_scale=am_scale,
        am_margin=am_margin,
    )
