"""Multi-task pre-training model on HuBERT backbone.

Trains per-dataset classification heads (PD detection, voice pathology) and
shared demographic heads (age regression, sex classification) on top of the
HuBERT backbone that will later be loaded into VoiceFM's AudioEncoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import HubertModel

from src.models.audio_encoder import AttentivePooling, SpecAugment


class HuBERTPretrainModel(nn.Module):
    """Multi-task pre-training model.

    Architecture:
        HuBERT (freeze layers 0..freeze_layers-1)
        -> SpecAugment (optional, training only)
        -> AttentivePooling (768 -> 768)
        -> Per-dataset disease heads: Linear(768, 2) each
        -> Shared age head: Linear(768, 1)
        -> Shared sex head: Linear(768, 3)  (male/female/other)
    """

    def __init__(
        self,
        backbone: str = "facebook/hubert-base-ls960",
        freeze_layers: int = 8,
        num_datasets: int = 2,
        spec_augment: bool = False,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()

        self.hubert = HubertModel.from_pretrained(backbone)
        hidden_dim: int = self.hubert.config.hidden_size  # 768

        if gradient_checkpointing:
            self.hubert.gradient_checkpointing_enable()

        # Freeze CNN feature extractor + projection
        for param in self.hubert.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.hubert.feature_projection.parameters():
            param.requires_grad = False

        # Freeze first N transformer layers
        for layer_idx in range(min(freeze_layers, len(self.hubert.encoder.layers))):
            for param in self.hubert.encoder.layers[layer_idx].parameters():
                param.requires_grad = False

        self.spec_augment = SpecAugment() if spec_augment else None
        self.pooling = AttentivePooling(hidden_dim)

        # Per-dataset binary classification heads
        self.disease_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 2) for _ in range(num_datasets)
        ])

        # Shared demographic heads
        self.age_head = nn.Linear(hidden_dim, 1)
        self.sex_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        audio_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        dataset_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            audio_values: (batch, samples) raw waveform at 16kHz.
            attention_mask: (batch, samples) binary mask.
            dataset_ids: (batch,) integer identifying which dataset each sample
                comes from (selects the appropriate disease head).

        Returns:
            Dict with keys: pooled, disease_logits, age_pred, sex_logits.
        """
        out = self.hubert(
            input_values=audio_values,
            attention_mask=attention_mask,
            output_attentions=False,
            return_dict=True,
        )
        hidden_states = out.last_hidden_state  # (B, T, H)

        if self.spec_augment is not None:
            hidden_states = self.spec_augment(hidden_states)

        # Frame-level mask
        frame_mask = None
        if attention_mask is not None:
            frame_mask = self.hubert._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask,
            )

        pooled = self.pooling(hidden_states, frame_mask)  # (B, H)

        # Per-dataset disease logits
        if dataset_ids is not None:
            disease_logits = torch.zeros(pooled.shape[0], 2, device=pooled.device)
            for ds_id in dataset_ids.unique():
                mask = dataset_ids == ds_id
                disease_logits[mask] = self.disease_heads[ds_id.item()](pooled[mask]).float()
        else:
            disease_logits = self.disease_heads[0](pooled).float()

        age_pred = self.age_head(pooled).squeeze(-1)  # (B,)
        sex_logits = self.sex_head(pooled)  # (B, 3)

        return {
            "pooled": pooled,
            "disease_logits": disease_logits,
            "age_pred": age_pred,
            "sex_logits": sex_logits,
        }

    def get_backbone_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract HuBERT + pooling weights for loading into AudioEncoder."""
        state = {}
        for key, val in self.state_dict().items():
            if key.startswith("hubert.") or key.startswith("pooling."):
                state[key] = val
        return state
