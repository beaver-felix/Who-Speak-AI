"""Numerically stable additive angular-margin Softmax for speaker training."""

from __future__ import annotations

import math
from typing import NamedTuple

try:
    import torch
    import torch.nn.functional as functional
except ModuleNotFoundError as error:  # pragma: no cover - Kaggle dependency gate.
    raise ModuleNotFoundError(
        "Training objectives require Kaggle's existing CUDA-matched PyTorch "
        "build; PyTorch is deliberately not installed by this project."
    ) from error


class AamSoftmaxOutput(NamedTuple):
    """Return the scalar loss, scaled logits, and batch accuracy."""

    loss: torch.Tensor
    logits: torch.Tensor
    accuracy: torch.Tensor


class AamSoftmaxHead(torch.nn.Module):
    """Apply AAM-Softmax to embeddings with a target-dataset class matrix.

    Angular operations run in float32 even under automatic mixed precision.
    This avoids half-precision cancellation near cosine values of plus or
    minus one while preserving gradients to the encoder and class weights.
    """

    def __init__(
        self,
        *,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.2,
        scale: float = 30.0,
        easy_margin: bool = False,
    ) -> None:
        """Create a new classifier and precompute fixed margin constants."""
        super().__init__()
        if (
            isinstance(embedding_dim, bool)
            or not isinstance(embedding_dim, int)
            or embedding_dim <= 0
        ):
            raise ValueError("embedding_dim must be a positive integer.")
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes <= 1
        ):
            raise ValueError("num_classes must be an integer greater than one.")
        if not 0.0 < margin < math.pi / 2.0:
            raise ValueError("margin must be in (0, pi/2).")
        if scale <= 0.0 or not math.isfinite(scale):
            raise ValueError("scale must be finite and positive.")
        if not isinstance(easy_margin, bool):
            raise ValueError("easy_margin must be boolean.")

        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.margin = float(margin)
        self.scale = float(scale)
        self.easy_margin = easy_margin
        self.weight = torch.nn.Parameter(
            torch.empty(self.num_classes, self.embedding_dim)
        )
        torch.nn.init.xavier_normal_(self.weight)

        self._cos_margin = math.cos(self.margin)
        self._sin_margin = math.sin(self.margin)
        self._threshold = math.cos(math.pi - self.margin)
        self._threshold_correction = math.sin(math.pi - self.margin) * self.margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> AamSoftmaxOutput:
        """Compute the mean AAM cross-entropy loss for one training batch."""
        self._validate_inputs(embeddings, labels)

        # AAM geometry is sensitive around |cos(theta)| = 1. Force float32
        # inside an autocast-disabled region while retaining autograd links.
        device_type = embeddings.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            normalized_embeddings = functional.normalize(
                embeddings.float(),
                p=2,
                dim=1,
            )
            normalized_weights = functional.normalize(
                self.weight.float(),
                p=2,
                dim=1,
            )
            cosine = functional.linear(
                normalized_embeddings,
                normalized_weights,
            ).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            target_cosine = cosine.gather(1, labels[:, None])
            sine = torch.sqrt((1.0 - target_cosine.square()).clamp_min(1e-7))
            target_margin_cosine = (
                target_cosine * self._cos_margin - sine * self._sin_margin
            )
            if self.easy_margin:
                target_margin_cosine = torch.where(
                    target_cosine > 0.0,
                    target_margin_cosine,
                    target_cosine,
                )
            else:
                target_margin_cosine = torch.where(
                    target_cosine > self._threshold,
                    target_margin_cosine,
                    target_cosine - self._threshold_correction,
                )

            # Scatter only the target column. This avoids allocating a dense
            # one-hot matrix for TidyVoice's and ViMD's thousands of speakers.
            logits = cosine.clone()
            logits.scatter_(1, labels[:, None], target_margin_cosine)
            logits = logits * self.scale
            loss = functional.cross_entropy(logits, labels, reduction="mean")
            accuracy = (logits.argmax(dim=1) == labels).float().mean()
        return AamSoftmaxOutput(loss=loss, logits=logits, accuracy=accuracy)

    def _validate_inputs(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Reject malformed batches before expensive class-matrix operations."""
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                "embeddings must have shape [batch, embedding_dim]."
            )
        if embeddings.shape[0] == 0:
            raise ValueError("embeddings must contain at least one item.")
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels must have shape [batch].")
        if labels.dtype != torch.long:
            raise ValueError("labels must use torch.int64 dtype.")
        if labels.device != embeddings.device or self.weight.device != embeddings.device:
            raise ValueError("embeddings, labels, and head must share one device.")
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError("embeddings must contain only finite values.")
        if bool((labels < 0).any()) or bool((labels >= self.num_classes).any()):
            raise ValueError("labels contain an out-of-range class index.")
