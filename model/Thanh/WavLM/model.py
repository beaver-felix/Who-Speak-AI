"""Hugging Face WavLM speaker-embedding wrapper.

The default path keeps the pretrained ``WavLMForXVector`` speaker head from
``microsoft/wavlm-base-plus-sv``. Mean and statistics pooling are available
for experiments that use the SSL transformer's hidden states directly.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as torch_functional

try:
    from transformers import AutoModelForAudioXVector
except ImportError as exc:  # pragma: no cover - depends on Kaggle environment.
    raise ImportError(
        "WavLM requires transformers with audio x-vector support; "
        "install it with `pip install transformers>=4.40`."
    ) from exc


class AdditiveMarginSoftmax(nn.Module):
    """Cosine classifier with an additive target-class margin."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.20,
    ) -> None:
        """Initialize trainable normalized class prototypes."""
        super().__init__()
        if num_classes < 2:
            raise ValueError("at least two speaker classes are required")
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        """Return scaled cosine logits with an optional additive margin."""
        embeddings = torch_functional.normalize(embeddings, p=2, dim=-1)
        weights = torch_functional.normalize(self.weight, p=2, dim=-1)
        cosine = embeddings @ weights.transpose(0, 1)
        if labels is not None:
            if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
                raise ValueError("labels must have shape [batch]")
            targets = torch.zeros_like(cosine)
            targets.scatter_(1, labels.long().unsqueeze(1), 1.0)
            cosine = cosine - targets * self.margin
        return cosine * self.scale


class WavLMSpeakerModel(nn.Module):
    """WavLM encoder with selectable pooling and an optional speaker head."""

    SUPPORTED_POOLING = {"xvector", "mean", "statistics"}

    def __init__(
        self,
        pretrained_model_name: str = "microsoft/wavlm-base-plus-sv",
        num_speakers: Optional[int] = None,
        pooling: str = "xvector",
        embedding_dim: int = 512,
        freeze_backbone: bool = True,
        unfreeze_last_n_layers: int = 0,
        gradient_checkpointing: bool = False,
        am_scale: float = 30.0,
        am_margin: float = 0.20,
    ) -> None:
        """Load the pretrained model and configure memory-efficient tuning."""
        super().__init__()
        pooling = pooling.lower()
        if pooling not in self.SUPPORTED_POOLING:
            raise ValueError(f"pooling must be one of {sorted(self.SUPPORTED_POOLING)}")
        if embedding_dim <= 0 or unfreeze_last_n_layers < 0:
            raise ValueError("embedding_dim must be positive and layer count non-negative")

        self.pretrained_model_name = pretrained_model_name
        self.pooling = pooling
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.freeze_backbone_requested = freeze_backbone
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.backbone = AutoModelForAudioXVector.from_pretrained(pretrained_model_name)

        hidden_size = int(self.backbone.config.hidden_size)
        native_xvector_dim = int(
            getattr(self.backbone.config, "xvector_output_dim", hidden_size)
        )
        pooled_dim = {
            "xvector": native_xvector_dim,
            "mean": hidden_size,
            "statistics": hidden_size * 2,
        }[pooling]
        self.projection = (
            nn.Identity()
            if pooled_dim == embedding_dim
            else nn.Sequential(nn.Linear(pooled_dim, embedding_dim), nn.LayerNorm(embedding_dim))
        )
        self.classifier = (
            AdditiveMarginSoftmax(embedding_dim, num_speakers, am_scale, am_margin)
            if num_speakers is not None
            else None
        )

        self._configure_backbone_trainability(freeze_backbone, unfreeze_last_n_layers)
        self.backbone_fully_frozen = not any(
            parameter.requires_grad for parameter in self.backbone.parameters()
        )
        if gradient_checkpointing and not self.backbone_fully_frozen:
            self.backbone.gradient_checkpointing_enable()

    def _configure_backbone_trainability(
        self, freeze_backbone: bool, unfreeze_last_n_layers: int
    ) -> None:
        """Freeze the backbone and optionally reopen its final transformer layers."""
        if not freeze_backbone:
            return
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if unfreeze_last_n_layers == 0:
            return

        encoder_layers = self.backbone.wavlm.encoder.layers
        if unfreeze_last_n_layers > len(encoder_layers):
            raise ValueError(
                f"cannot unfreeze {unfreeze_last_n_layers} layers; "
                f"the backbone has {len(encoder_layers)}"
            )
        for layer in encoder_layers[-unfreeze_last_n_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        # The final encoder normalization should adapt with reopened layers.
        if hasattr(self.backbone.wavlm.encoder, "layer_norm"):
            for parameter in self.backbone.wavlm.encoder.layer_norm.parameters():
                parameter.requires_grad = True

    def train(self, mode: bool = True) -> "WavLMSpeakerModel":
        """Keep a fully frozen backbone deterministic during head training."""
        super().train(mode)
        if mode and self.backbone_fully_frozen:
            self.backbone.eval()
        return self

    def _feature_attention_mask(
        self, attention_mask: Tensor, feature_length: int
    ) -> Tensor:
        """Convert a sample-level attention mask to the encoded frame rate."""
        if hasattr(self.backbone, "_get_feature_vector_attention_mask"):
            return self.backbone._get_feature_vector_attention_mask(
                feature_length, attention_mask
            )
        return (
            torch_functional.interpolate(
                attention_mask[:, None].float(), size=feature_length, mode="nearest"
            )[:, 0]
            .to(dtype=torch.bool)
        )

    @staticmethod
    def _masked_statistics(features: Tensor, mask: Optional[Tensor]) -> tuple[Tensor, Tensor]:
        """Calculate masked temporal mean and standard deviation."""
        if mask is None:
            return features.mean(dim=1), features.std(dim=1, unbiased=False)
        weights = mask.to(dtype=features.dtype).unsqueeze(-1)
        count = weights.sum(dim=1).clamp_min(1.0)
        mean = (features * weights).sum(dim=1) / count
        variance = ((features - mean.unsqueeze(1)).square() * weights).sum(dim=1) / count
        return mean, variance.clamp_min(1e-9).sqrt()

    def _encode_backbone(
        self, waveform: Tensor, attention_mask: Optional[Tensor]
    ) -> Tensor:
        """Run either the pretrained x-vector path or custom SSL pooling."""
        if self.pooling == "xvector":
            output = self.backbone(
                input_values=waveform,
                attention_mask=attention_mask,
                return_dict=True,
            )
            return output.embeddings

        hidden_output = self.backbone.wavlm(
            input_values=waveform,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        frame_mask = (
            self._feature_attention_mask(attention_mask, hidden_output.shape[1])
            if attention_mask is not None
            else None
        )
        mean, standard_deviation = self._masked_statistics(hidden_output, frame_mask)
        if self.pooling == "mean":
            return mean
        return torch.cat((mean, standard_deviation), dim=-1)

    def encode(self, waveform: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Produce unnormalized fixed-dimensional speaker embeddings."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [time] or [batch, time]")
        if attention_mask is not None and attention_mask.shape != waveform.shape:
            raise ValueError("attention_mask must match the waveform shape")
        waveform = waveform.float()
        context = torch.no_grad() if self.backbone_fully_frozen else nullcontext()
        with context:
            pooled = self._encode_backbone(waveform, attention_mask)
        return self.projection(pooled)

    def forward(
        self,
        waveform: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Return embeddings, or AM-Softmax logits when labels are provided."""
        embeddings = self.encode(waveform, attention_mask)
        if labels is not None:
            if self.classifier is None:
                raise RuntimeError("classification head was not created")
            return self.classifier(embeddings, labels)
        return embeddings


def extract_speaker_embedding(
    model: nn.Module,
    waveform: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """Return an L2-normalized embedding suitable for cosine similarity."""
    was_single = waveform.ndim == 1
    if was_single:
        waveform = waveform.unsqueeze(0)
        if attention_mask is not None and attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
    device = next(model.parameters()).device
    waveform = waveform.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.no_grad():
        embeddings = model(waveform, attention_mask=attention_mask)
    embeddings = torch_functional.normalize(embeddings.float(), p=2, dim=-1)
    return embeddings[0] if was_single else embeddings


def build_model(
    pretrained_model_name: str = "microsoft/wavlm-base-plus-sv",
    num_speakers: Optional[int] = None,
    pooling: str = "xvector",
    embedding_dim: int = 512,
    freeze_backbone: bool = True,
    unfreeze_last_n_layers: int = 0,
    gradient_checkpointing: bool = False,
    am_scale: float = 30.0,
    am_margin: float = 0.20,
) -> WavLMSpeakerModel:
    """Construct a checkpointable WavLM speaker model."""
    return WavLMSpeakerModel(
        pretrained_model_name=pretrained_model_name,
        num_speakers=num_speakers,
        pooling=pooling,
        embedding_dim=embedding_dim,
        freeze_backbone=freeze_backbone,
        unfreeze_last_n_layers=unfreeze_last_n_layers,
        gradient_checkpointing=gradient_checkpointing,
        am_scale=am_scale,
        am_margin=am_margin,
    )
