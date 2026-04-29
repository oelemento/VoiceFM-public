"""
HuBERT-based audio encoder for VoiceFM.

Extracts frame-level features from raw waveforms using a pretrained HuBERT backbone,
applies attentive pooling to produce a fixed-size vector, and projects into a shared
embedding space for contrastive learning with clinical profiles.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel


class SpecAugment(nn.Module):
    """SpecAugment-style masking on frame-level features.

    Applies random time and frequency masking to the HuBERT frame-level
    representations during training.

    Args:
        freq_mask_param: Maximum number of frequency (feature) channels to mask.
        time_mask_param: Maximum number of time steps to mask.
        num_freq_masks: Number of frequency masks to apply.
        num_time_masks: Number of time masks to apply.
    """

    def __init__(
        self,
        freq_mask_param: int = 48,
        time_mask_param: int = 40,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ) -> None:
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment masking.

        Args:
            x: ``(batch, seq_len, hidden_dim)`` frame-level features.

        Returns:
            Masked features of the same shape.
        """
        if not self.training:
            return x

        x = x.clone()
        _, T, F_dim = x.shape

        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            f = min(f, F_dim)
            if f > 0:
                f0 = torch.randint(0, F_dim - f + 1, (1,)).item()
                x[:, :, f0 : f0 + f] = 0.0

        for _ in range(self.num_time_masks):
            t = torch.randint(0, self.time_mask_param + 1, (1,)).item()
            t = min(t, T)
            if t > 0:
                t0 = torch.randint(0, T - t + 1, (1,)).item()
                x[:, t0 : t0 + t, :] = 0.0

        return x


class AttentivePooling(nn.Module):
    """Attentive pooling over frame-level features to produce a fixed-size vector.

    Learns a query vector that attends over temporal positions, producing a
    weighted sum of frame-level representations.

    Args:
        hidden_dim: Dimensionality of input frame features.
    """

    def __init__(self, hidden_dim: int = 768) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention-weighted pooling over temporal frames.

        Args:
            x: Frame-level features of shape ``(batch, seq_len, hidden_dim)``.
            attention_mask: Binary mask of shape ``(batch, seq_len)`` where ``1``
                indicates valid frames and ``0`` indicates padding. If ``None``,
                all frames are treated as valid.

        Returns:
            Pooled representation of shape ``(batch, hidden_dim)``.
        """
        # (batch, seq_len, 1)
        attn_scores = self.attention(x)

        if attention_mask is not None:
            # Expand mask to match scores: (batch, seq_len) -> (batch, seq_len, 1)
            mask = attention_mask.unsqueeze(-1)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)
        pooled = (attn_weights * x).sum(dim=1)  # (batch, hidden_dim)
        return pooled


class AudioEncoder(nn.Module):
    """HuBERT backbone with attentive pooling and a projection head.

    Processes raw 16 kHz waveforms through a pretrained HuBERT model, optionally
    adds task-type embeddings, pools across time with learned attention, and projects
    into the shared contrastive embedding space.

    Args:
        backbone: HuggingFace model identifier for the HuBERT checkpoint.
        freeze_layers: Number of transformer layers to freeze (layers 0 through
            ``freeze_layers - 1`` plus the feature extractor are frozen).
        projection_dim: Output embedding dimensionality.
        num_task_types: Number of distinct recording task types for the
            task-type embedding table.
    """

    def __init__(
        self,
        backbone: str = "facebook/hubert-base-ls960",
        freeze_layers: int = 8,
        projection_dim: int = 256,
        num_task_types: int = 50,
        spec_augment: bool = False,
        gradient_checkpointing: bool = False,
        task_conditioning: str = "additive",
    ) -> None:
        super().__init__()

        self.backbone_name = backbone
        self.freeze_layers = freeze_layers
        self.projection_dim = projection_dim
        self.use_spec_augment = spec_augment
        self.task_conditioning = task_conditioning

        # --- HuBERT backbone ---
        self.hubert = HubertModel.from_pretrained(backbone)

        # Gradient checkpointing trades compute for memory
        if gradient_checkpointing:
            self.hubert.gradient_checkpointing_enable()
        hidden_dim: int = self.hubert.config.hidden_size  # 768 for base

        # Freeze the CNN feature extractor
        for param in self.hubert.feature_extractor.parameters():
            param.requires_grad = False
        # Also freeze feature_projection which sits between CNN and transformer
        for param in self.hubert.feature_projection.parameters():
            param.requires_grad = False

        # Freeze the first `freeze_layers` transformer layers
        for layer_idx in range(min(freeze_layers, len(self.hubert.encoder.layers))):
            for param in self.hubert.encoder.layers[layer_idx].parameters():
                param.requires_grad = False

        # --- Task-type conditioning ---
        if task_conditioning == "film":
            # FiLM: output scale + shift (2 * hidden_dim)
            self.task_embedding = nn.Embedding(num_task_types, 2 * hidden_dim)
        else:
            self.task_embedding = nn.Embedding(num_task_types, hidden_dim)

        # --- SpecAugment (optional) ---
        self.spec_augment = SpecAugment() if spec_augment else None

        # --- Attentive pooling ---
        self.pooling = AttentivePooling(hidden_dim)

        # --- Projection head: 768 -> 512 -> 256 ---
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, projection_dim),
        )

    def forward(
        self,
        audio_input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        task_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode raw waveforms into L2-normalized embeddings.

        Args:
            audio_input_values: Raw waveform tensor of shape ``(batch, samples)``
                sampled at 16 kHz.
            attention_mask: Binary mask of shape ``(batch, samples)`` indicating
                valid samples (useful for variable-length inputs in a batch).
            task_type_ids: Integer tensor of shape ``(batch,)`` identifying the
                recording task type for each sample.

        Returns:
            L2-normalized embeddings of shape ``(batch, projection_dim)``.
        """
        # Extract frame-level features from HuBERT
        hubert_output = self.hubert(
            input_values=audio_input_values,
            attention_mask=attention_mask,
            output_attentions=False,
            return_dict=True,
        )

        # (batch, seq_len, hidden_dim)
        hidden_states = hubert_output.last_hidden_state

        # SpecAugment on frame-level features (training only)
        if self.spec_augment is not None:
            hidden_states = self.spec_augment(hidden_states)

        # Add task-type conditioning if provided
        if task_type_ids is not None:
            if self.task_conditioning == "film":
                # FiLM: feature-wise scale + shift
                film_params = self.task_embedding(task_type_ids)  # (B, 2*H)
                scale, shift = film_params.chunk(2, dim=-1)  # each (B, H)
                scale = scale.unsqueeze(1)  # (B, 1, H) for broadcasting
                shift = shift.unsqueeze(1)
                hidden_states = hidden_states * (1 + scale) + shift
            else:
                # Additive: simple bias
                task_emb = self.task_embedding(task_type_ids).unsqueeze(1)
                hidden_states = hidden_states + task_emb

        # Build frame-level attention mask from the HuBERT output mask.
        # HuBERT downsamples via the CNN, so we need the reduced-length mask
        # that the model produces internally.
        frame_mask: torch.Tensor | None = None
        if attention_mask is not None:
            # Derive the output-length mask from HuBERT's feature extractor
            frame_mask = self.hubert._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )

        # Attentive pooling -> (batch, hidden_dim)
        pooled = self.pooling(hidden_states, frame_mask)

        # Project and L2-normalize
        embeddings = self.projection(pooled)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings
