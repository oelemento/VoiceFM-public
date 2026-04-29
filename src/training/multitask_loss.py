"""Multi-task loss for clinical voice foundation model.

Computes weighted sum of per-task losses:
- Binary tasks: BCE with logits
- Regression tasks: MSE with NaN masking
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """Weighted multi-task loss with NaN masking for missing labels.

    Args:
        task_configs: List of task dicts with {name, type, input_key, weight}.
            type is 'binary' or 'regression'.
        regression_stats: Optional dict of {input_key: {"mean": float, "std": float}}
            for z-score normalizing regression targets before MSE. Keys that are
            already normalized in the dataset (e.g. 'age') should NOT be included.
    """

    def __init__(self, task_configs: list[dict], regression_stats: dict | None = None) -> None:
        super().__init__()
        self.task_configs = task_configs
        self.regression_stats = regression_stats or {}

    def forward(
        self,
        task_outputs: dict[str, torch.Tensor],
        clinical_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute per-task losses and weighted total.

        Args:
            task_outputs: {task_name: (batch, 1)} from model forward.
            clinical_features: {feature_name: (batch,) or (batch, K)} from batch.

        Returns:
            Dict with 'total_loss' and per-task '{name}_loss' entries.
        """
        losses = {}
        total = torch.tensor(0.0, device=next(iter(task_outputs.values())).device, requires_grad=True)

        for task in self.task_configs:
            name = task["name"]
            input_key = task["input_key"]
            task_type = task["type"]
            weight = task.get("weight", 1.0)

            preds = task_outputs[name].squeeze(-1)  # (batch,)
            targets = clinical_features.get(input_key)

            if targets is None:
                losses[f"{name}_loss"] = torch.tensor(0.0, device=preds.device)
                continue

            targets = targets.float().squeeze(-1)  # (batch,)

            if task_type == "binary":
                # BCE with logits — mask out invalid targets (NaN)
                valid = ~torch.isnan(targets)
                if valid.any():
                    loss = F.binary_cross_entropy_with_logits(
                        preds[valid], targets[valid],
                    )
                else:
                    loss = torch.tensor(0.0, device=preds.device, requires_grad=True)

            elif task_type == "regression":
                # MSE with NaN masking
                valid = ~torch.isnan(targets)
                if valid.any():
                    t = targets[valid]
                    # Z-score normalize targets if stats provided (e.g. questionnaire scores)
                    stats = self.regression_stats.get(input_key)
                    if stats:
                        t = (t - stats["mean"]) / stats["std"]
                    loss = F.mse_loss(preds[valid], t)
                else:
                    loss = torch.tensor(0.0, device=preds.device, requires_grad=True)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            losses[f"{name}_loss"] = loss
            total = total + weight * loss

        losses["total_loss"] = total
        return losses
