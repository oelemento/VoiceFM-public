"""Multi-task supervised clinical voice foundation model.

Fine-tunes HuBERT with multiple clinical task heads (binary classification
and regression) to learn a general-purpose voice representation.
The shared backbone becomes the foundation model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel

from src.models.audio_encoder import SpecAugment, AttentivePooling


class ClinicalVoiceModel(nn.Module):
    """HuBERT backbone with multi-task clinical prediction heads.

    Architecture:
        HuBERT-base (partial freeze) → SpecAugment → task-type embedding
        → AttentivePooling → 768-dim foundation embedding → task heads

    Args:
        config: Dict with keys:
            backbone.name: HuggingFace model identifier
            backbone.freeze_layers: Number of transformer layers to freeze
            backbone.gradient_checkpointing: Enable gradient checkpointing
            spec_augment: Whether to apply SpecAugment
            task_conditioning.num_task_types: Number of recording task types
            tasks: List of task dicts with {name, type, input_key, weight}
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        backbone_cfg = config["backbone"]
        self.backbone_name = backbone_cfg["name"]
        self.freeze_layers = backbone_cfg["freeze_layers"]

        # --- HuBERT backbone ---
        self.hubert = HubertModel.from_pretrained(self.backbone_name)

        if backbone_cfg.get("gradient_checkpointing", False):
            self.hubert.gradient_checkpointing_enable()

        hidden_dim: int = self.hubert.config.hidden_size  # 768 for base

        # Freeze CNN feature extractor + feature projection
        for param in self.hubert.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.hubert.feature_projection.parameters():
            param.requires_grad = False

        # Freeze first N transformer layers
        for layer_idx in range(min(self.freeze_layers, len(self.hubert.encoder.layers))):
            for param in self.hubert.encoder.layers[layer_idx].parameters():
                param.requires_grad = False

        # --- Task-type conditioning ---
        num_task_types = config.get("task_conditioning", {}).get("num_task_types", 80)
        self.task_embedding = nn.Embedding(num_task_types, hidden_dim)

        # --- SpecAugment (optional) ---
        self.spec_augment = SpecAugment() if config.get("spec_augment", False) else None

        # --- Attentive pooling → 768-dim foundation embedding ---
        self.pooling = AttentivePooling(hidden_dim)

        # --- Task heads ---
        self.task_configs = config.get("tasks", [])
        self.task_heads = nn.ModuleDict()
        for task in self.task_configs:
            name = task["name"]
            # All heads: Linear(768, 1)
            self.task_heads[name] = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        audio_input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        task_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict]:
        """Forward pass through backbone and all task heads.

        Args:
            audio_input_values: Raw waveform (batch, samples) at 16kHz.
            attention_mask: Binary mask (batch, samples).
            task_type_ids: Integer tensor (batch,) for recording task type.

        Returns:
            Dict with:
                pooled: Foundation embedding (batch, 768)
                task_outputs: {task_name: (batch, 1)} predictions
        """
        hubert_output = self.hubert(
            input_values=audio_input_values,
            attention_mask=attention_mask,
            output_attentions=False,
            return_dict=True,
        )

        hidden_states = hubert_output.last_hidden_state  # (batch, seq_len, 768)

        # SpecAugment on frame-level features (training only)
        if self.spec_augment is not None:
            hidden_states = self.spec_augment(hidden_states)

        # Add task-type embedding
        if task_type_ids is not None:
            task_emb = self.task_embedding(task_type_ids).unsqueeze(1)
            hidden_states = hidden_states + task_emb

        # Frame-level attention mask from HuBERT's CNN downsampling
        frame_mask: torch.Tensor | None = None
        if attention_mask is not None:
            frame_mask = self.hubert._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )

        # Attentive pooling → (batch, 768) foundation embedding
        pooled = self.pooling(hidden_states, frame_mask)

        # Task head predictions
        task_outputs = {}
        for task in self.task_configs:
            name = task["name"]
            task_outputs[name] = self.task_heads[name](pooled)  # (batch, 1)

        return {"pooled": pooled, "task_outputs": task_outputs}

    def get_foundation_state_dict(self) -> dict:
        """Return state dict containing only the foundation model weights.

        Includes HuBERT backbone, task embedding, and attentive pooling.
        Excludes task heads (they're discarded after training).
        """
        foundation_keys = {"hubert.", "task_embedding.", "pooling."}
        return {
            k: v for k, v in self.state_dict().items()
            if any(k.startswith(prefix) for prefix in foundation_keys)
        }

    @torch.no_grad()
    def extract_embeddings(
        self,
        audio_input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        task_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract 768-dim foundation embeddings (for downstream eval).

        Returns:
            L2-normalized pooled representation (batch, 768).
        """
        self.eval()
        output = self.forward(audio_input_values, attention_mask, task_type_ids)
        return F.normalize(output["pooled"], p=2, dim=-1)
