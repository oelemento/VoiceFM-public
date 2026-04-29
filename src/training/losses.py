"""
Loss functions for VoiceFM contrastive training.

Provides:
- InfoNCELoss: symmetric contrastive loss (CLIP-style) with optional memory
  queue for additional negatives and multi-positive masking.
- VoiceFMLoss: combined training objective with contrastive loss plus
  auxiliary supervision for disease category and age prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLIPLoss(nn.Module):
    """Sigmoid pairwise contrastive loss (SigLIP).

    Treats each (audio_i, clinical_j) pair independently with binary
    cross-entropy. Positive pairs push similarity up, negatives push it
    down. No softmax normalization avoids competition between negatives,
    letting subtle clinical signals survive.

    Reference: Zhai et al., "Sigmoid Loss for Language Image Pre-Training", ICCV 2023.
    """

    def __init__(self) -> None:
        super().__init__()
        # Learnable bias for decision boundary calibration
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        audio_embeds: torch.Tensor,
        clinical_embeds: torch.Tensor,
        temperature: torch.Tensor,
        audio_queue: torch.Tensor | None = None,
        clinical_queue: torch.Tensor | None = None,
        participant_ids: list[str] | None = None,
    ) -> torch.Tensor:
        B = audio_embeds.shape[0]

        # Pairwise similarity matrix: (B, B)
        logits = (audio_embeds @ clinical_embeds.T) / temperature + self.bias

        # Target matrix: +1 for positive pairs, -1 for negative pairs
        if participant_ids is not None:
            targets = -torch.ones(B, B, device=logits.device)
            for i, pid_i in enumerate(participant_ids):
                for j, pid_j in enumerate(participant_ids):
                    if pid_i == pid_j:
                        targets[i, j] = 1
        else:
            targets = -torch.ones(B, B, device=logits.device)
            targets.fill_diagonal_(1)

        # Sigmoid loss: -log(sigmoid(target * logit))
        # Equivalent to log(1 + exp(-target * logit))
        return -F.logsigmoid(targets * logits).mean()


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE loss with memory queue and hard negative support.

    Supports extra negatives from a memory queue (MoCo-style),
    multi-positive masking when multiple recordings share a participant,
    and hard negative mining with configurable weighting.

    Args:
        hard_negative_mining: If True, upweight harder negatives in the loss.
        hard_negative_beta: Concentration parameter for hard negative weighting.
            Higher values weight harder negatives more strongly. Default 1.0.
    """

    def __init__(
        self,
        hard_negative_mining: bool = False,
        hard_negative_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.hard_negative_mining = hard_negative_mining
        self.hard_negative_beta = hard_negative_beta

    def forward(
        self,
        audio_embeds: torch.Tensor,
        clinical_embeds: torch.Tensor,
        temperature: torch.Tensor,
        audio_queue: torch.Tensor | None = None,
        clinical_queue: torch.Tensor | None = None,
        participant_ids: list[str] | None = None,
    ) -> torch.Tensor:
        B = audio_embeds.shape[0]

        # Build positive mask for multi-recording batches
        # mask[i,j] = True if recording i and clinical j are from same participant
        if participant_ids is not None:
            pos_mask = torch.zeros(B, B, dtype=torch.bool, device=audio_embeds.device)
            for i, pid_i in enumerate(participant_ids):
                for j, pid_j in enumerate(participant_ids):
                    if pid_i == pid_j:
                        pos_mask[i, j] = True
        else:
            pos_mask = None

        # --- Audio -> Clinical direction ---
        if clinical_queue is not None and clinical_queue.shape[0] > 0:
            all_clinical = torch.cat([clinical_embeds, clinical_queue.to(audio_embeds.device)])
        else:
            all_clinical = clinical_embeds
        logits_a2c = (audio_embeds @ all_clinical.T) / temperature

        # --- Clinical -> Audio direction ---
        if audio_queue is not None and audio_queue.shape[0] > 0:
            all_audio = torch.cat([audio_embeds, audio_queue.to(clinical_embeds.device)])
        else:
            all_audio = audio_embeds
        logits_c2a = (clinical_embeds @ all_audio.T) / temperature

        if pos_mask is not None and pos_mask.sum() > B:
            # Multi-positive: use soft cross-entropy
            # Targets are uniform over all positives for each row
            # Extend mask for queue columns (all negatives)
            Q_c = logits_a2c.shape[1] - B
            if Q_c > 0:
                mask_a2c = torch.cat([pos_mask, torch.zeros(B, Q_c, dtype=torch.bool,
                                      device=audio_embeds.device)], dim=1)
            else:
                mask_a2c = pos_mask
            targets_a2c = mask_a2c.float() / mask_a2c.float().sum(dim=1, keepdim=True)
            loss_a2c = (-targets_a2c * F.log_softmax(logits_a2c, dim=1)).sum(dim=1).mean()

            Q_a = logits_c2a.shape[1] - B
            if Q_a > 0:
                mask_c2a = torch.cat([pos_mask.T, torch.zeros(B, Q_a, dtype=torch.bool,
                                      device=clinical_embeds.device)], dim=1)
            else:
                mask_c2a = pos_mask.T
            targets_c2a = mask_c2a.float() / mask_c2a.float().sum(dim=1, keepdim=True)
            loss_c2a = (-targets_c2a * F.log_softmax(logits_c2a, dim=1)).sum(dim=1).mean()
        elif self.hard_negative_mining:
            # Hard negative mining: upweight harder negatives
            # Weight each negative by exp(beta * sim), then normalize
            loss_a2c = self._hard_negative_ce(logits_a2c, B)
            loss_c2a = self._hard_negative_ce(logits_c2a, B)
        else:
            # Standard: diagonal is positive
            labels = torch.arange(B, device=audio_embeds.device)
            loss_a2c = F.cross_entropy(logits_a2c, labels)
            loss_c2a = F.cross_entropy(logits_c2a, labels)

        return (loss_a2c + loss_c2a) / 2.0

    def _hard_negative_ce(self, logits: torch.Tensor, B: int) -> torch.Tensor:
        """Cross-entropy with hard negative importance weighting.

        For each anchor, computes negative weights proportional to
        exp(beta * logit), concentrating the loss on harder negatives.
        """
        device = logits.device
        labels = torch.arange(B, device=device)

        # Create negative mask (everything except the diagonal positive)
        neg_mask = ~torch.eye(B, logits.shape[1], dtype=torch.bool, device=device)

        # Compute importance weights for negatives: w_j = exp(beta * s_j)
        neg_logits = logits.clone()
        neg_logits[~neg_mask] = float("-inf")  # zero out positives
        neg_weights = (self.hard_negative_beta * neg_logits).softmax(dim=1)

        # Reweight logits: multiply negative logits by importance weights
        # This amplifies harder negatives in the softmax denominator
        weighted_logits = logits.clone()
        neg_contribution = neg_weights * neg_mask.float()
        # Scale negatives: keep positive logit unchanged, scale negatives by weight * N_neg
        N_neg = neg_mask.sum(dim=1, keepdim=True).float()
        weighted_logits = logits * (~neg_mask).float() + logits * neg_contribution * N_neg

        return F.cross_entropy(weighted_logits, labels)


class VoiceFMLoss(nn.Module):
    """Combined VoiceFM training loss.

    Aggregates the contrastive InfoNCE loss with auxiliary losses for
    multi-label disease category classification (BCE) and age regression (MSE).
    """

    def __init__(
        self,
        disease_weight: float = 0.1,
        age_weight: float = 0.05,
        hard_negative_mining: bool = False,
        hard_negative_beta: float = 1.0,
        contrastive_loss_type: str = "infonce",
        site_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.disease_weight = disease_weight
        self.age_weight = age_weight
        self.site_weight = site_weight
        if contrastive_loss_type == "siglip":
            self.infonce = SigLIPLoss()
        else:
            self.infonce = InfoNCELoss(
                hard_negative_mining=hard_negative_mining,
                hard_negative_beta=hard_negative_beta,
            )

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        disease_targets: torch.Tensor,
        age_targets: torch.Tensor,
        audio_queue: torch.Tensor | None = None,
        clinical_queue: torch.Tensor | None = None,
        participant_ids: list[str] | None = None,
        site_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # --- Contrastive loss (with queue + multi-positive support) ---
        infonce_loss = self.infonce(
            model_output["audio_embeds"],
            model_output["clinical_embeds"],
            model_output["temperature"],
            audio_queue=audio_queue,
            clinical_queue=clinical_queue,
            participant_ids=participant_ids,
        )

        # --- Disease category loss (multi-label BCE) ---
        disease_loss = F.binary_cross_entropy_with_logits(
            model_output["disease_logits"],
            disease_targets.float(),
        )

        # --- Age regression loss (MSE with NaN masking) ---
        age_pred = model_output["age_pred"].squeeze(-1)
        age_targets_flat = age_targets.float().squeeze(-1)

        valid_age_mask = ~torch.isnan(age_targets_flat)

        if valid_age_mask.any():
            age_loss = F.mse_loss(
                age_pred[valid_age_mask],
                age_targets_flat[valid_age_mask],
            )
        else:
            age_loss = torch.tensor(0.0, device=age_pred.device, requires_grad=True)

        # --- Site adversarial loss (DANN) ---
        site_loss = torch.tensor(0.0, device=age_pred.device)
        if (
            self.site_weight > 0
            and site_targets is not None
            and "site_logits" in model_output
        ):
            valid_site_mask = site_targets >= 0
            if valid_site_mask.any():
                site_loss = F.cross_entropy(
                    model_output["site_logits"][valid_site_mask],
                    site_targets[valid_site_mask],
                )

        # --- Combined loss ---
        total_loss = (
            infonce_loss
            + self.disease_weight * disease_loss
            + self.age_weight * age_loss
            + self.site_weight * site_loss
        )

        return {
            "total_loss": total_loss,
            "infonce_loss": infonce_loss,
            "disease_loss": disease_loss,
            "age_loss": age_loss,
            "site_loss": site_loss,
        }
