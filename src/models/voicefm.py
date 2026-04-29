"""
VoiceFM: CLIP-style contrastive model for voice + clinical data.

Combines a HuBERT-based audio encoder with a tabular transformer clinical
encoder, aligning their representations via contrastive learning in a shared
embedding space. Includes auxiliary prediction heads for disease category
classification and age regression.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clinical_encoder import ClinicalEncoder


class VoiceFM(nn.Module):
    """CLIP-style contrastive model for voice recordings and clinical profiles.

    Produces aligned embeddings from both modalities and computes a scaled
    cosine similarity matrix for contrastive training. Auxiliary heads predict
    disease category and age from the audio embeddings.

    Args:
        audio_encoder: Audio encoder module (HuBERT or HeAR). Must expose
            ``projection_dim`` attribute and accept ``(audio_input_values,
            attention_mask, task_type_ids)`` in forward.
        clinical_encoder: The clinical encoder module.
        temperature_init: Initial value for the temperature parameter
            (applied as ``exp(log_temperature)``).
        learn_temperature: Whether the temperature is a learnable parameter.
        site_adversarial: If True, add a gradient-reversal site classifier.
        num_sites: Number of collection sites for the site classifier.
    """

    def __init__(
        self,
        audio_encoder: nn.Module,
        clinical_encoder: ClinicalEncoder,
        temperature_init: float = 0.07,
        learn_temperature: bool = True,
        site_adversarial: bool = False,
        num_sites: int = 6,
    ) -> None:
        super().__init__()

        self.audio_encoder = audio_encoder
        self.clinical_encoder = clinical_encoder
        self.site_adversarial = site_adversarial

        # Learnable log-temperature (log scale for numerical stability)
        log_temp = torch.tensor(temperature_init).log()
        if learn_temperature:
            self.log_temperature = nn.Parameter(log_temp)
        else:
            self.register_buffer("log_temperature", log_temp)

        # Auxiliary heads (operate on audio embeddings)
        projection_dim = audio_encoder.projection_dim
        self.disease_category_head = nn.Linear(projection_dim, 4)
        self.age_head = nn.Linear(projection_dim, 1)

        # Site-adversarial head (DANN)
        if site_adversarial:
            from .gradient_reversal import GRL
            self.grl = GRL()
            self.site_classifier = nn.Sequential(
                nn.Linear(projection_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_sites),
            )

    @property
    def temperature(self) -> torch.Tensor:
        """Current temperature value (always positive via exp)."""
        return self.log_temperature.exp()

    def forward(
        self,
        audio_input_values: torch.Tensor,
        attention_mask: torch.Tensor | None,
        task_type_ids: torch.Tensor | None,
        clinical_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run forward pass through both encoders and compute similarity matrices.

        Args:
            audio_input_values: Raw waveform tensor ``(batch, samples)`` at 16 kHz.
            attention_mask: Binary mask ``(batch, samples)`` for audio padding.
            task_type_ids: Integer tensor ``(batch,)`` of recording task types.
            clinical_features: Dictionary of clinical feature tensors.

        Returns:
            Dictionary containing:
                - ``audio_embeds``: ``(B, D)`` L2-normalized audio embeddings.
                - ``clinical_embeds``: ``(B, D)`` L2-normalized clinical embeddings.
                - ``logits_per_audio``: ``(B, B)`` temperature-scaled cosine
                  similarity (audio as queries, clinical as keys).
                - ``logits_per_clinical``: ``(B, B)`` transposed similarity matrix.
                - ``disease_logits``: ``(B, 4)`` disease category logits.
                - ``age_pred``: ``(B, 1)`` predicted age.
                - ``temperature``: Scalar temperature value.
        """
        # Encode both modalities -> L2-normalized embeddings
        audio_embeds = self.audio_encoder(
            audio_input_values=audio_input_values,
            attention_mask=attention_mask,
            task_type_ids=task_type_ids,
        )
        clinical_embeds = self.clinical_encoder(clinical_features)

        # Cosine similarity matrix scaled by learned temperature
        # Since embeddings are already L2-normalized, dot product = cosine similarity
        temperature = self.temperature
        logits_per_audio = (audio_embeds @ clinical_embeds.T) / temperature
        logits_per_clinical = logits_per_audio.T

        # Auxiliary predictions from audio embeddings
        disease_logits = self.disease_category_head(audio_embeds)
        age_pred = self.age_head(audio_embeds)

        output = {
            "audio_embeds": audio_embeds,
            "clinical_embeds": clinical_embeds,
            "logits_per_audio": logits_per_audio,
            "logits_per_clinical": logits_per_clinical,
            "disease_logits": disease_logits,
            "age_pred": age_pred,
            "temperature": temperature,
        }

        # Site-adversarial prediction (gradient-reversed)
        if self.site_adversarial:
            reversed_embeds = self.grl(audio_embeds)
            output["site_logits"] = self.site_classifier(reversed_embeds)

        return output
