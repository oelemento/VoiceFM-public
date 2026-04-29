"""Training loop for multi-task clinical voice foundation model."""

import time
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class MultiTaskTrainer:
    """Trains ClinicalVoiceModel with multi-task supervision.

    Simplified from VoiceFMTrainer: no contrastive loss machinery
    (no EmbeddingQueue, MomentumEncoder, retrieval metrics, temperature).

    Args:
        model: ClinicalVoiceModel instance
        loss_fn: MultiTaskLoss instance
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: Optimizer
        scheduler: LR scheduler (optional)
        device: torch device
        config: Training config dict
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: torch.device = torch.device("cpu"),
        config: dict | None = None,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}

        self.mixed_precision = self.config.get("mixed_precision", True)
        self.scaler = GradScaler(enabled=self.mixed_precision)
        self.grad_accum_steps = self.config.get("gradient_accumulation_steps", 1)

        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.patience = self.config.get("early_stopping_patience", 15)

        # Task configs for validation metric computation
        self.task_configs = self.config.get("task_configs", [])

        # Regression normalization stats (for consistent RMSE reporting)
        self.regression_stats = self.config.get("regression_stats", {})

        self._wandb = None

    def _init_wandb(self, project: str = "voicefm-multitask"):
        try:
            import wandb
            self._wandb = wandb
            wandb.init(project=project, config=self.config)
            wandb.watch(self.model, log_freq=100)
        except ImportError:
            logger.warning("wandb not installed, skipping logging")

    def _move_batch_to_device(self, batch: dict) -> dict:
        """Move batch tensors to device."""
        device_batch = {}
        device_batch["audio_input_values"] = batch["audio_input_values"].to(self.device)
        device_batch["attention_mask"] = batch["attention_mask"].to(self.device)
        device_batch["task_type_ids"] = batch["task_type_id"].to(self.device)

        # Clinical features are dicts of tensors
        clinical = {}
        for key, val in batch["clinical_features"].items():
            if isinstance(val, torch.Tensor):
                clinical[key] = val.to(self.device)
        device_batch["clinical_features"] = clinical

        # Pass through string metadata
        device_batch["participant_id"] = batch.get("participant_id", [])

        return device_batch

    def train_epoch(self, epoch: int) -> dict:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        n_batches = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            batch = self._move_batch_to_device(batch)

            with autocast(device_type=self.device.type, enabled=self.mixed_precision):
                output = self.model(
                    audio_input_values=batch["audio_input_values"],
                    attention_mask=batch["attention_mask"],
                    task_type_ids=batch["task_type_ids"],
                )

                losses = self.loss_fn(
                    output["task_outputs"],
                    batch["clinical_features"],
                )
                scaled_loss = losses["total_loss"] / self.grad_accum_steps

            self.scaler.scale(scaled_loss).backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += losses["total_loss"].item()
            for k, v in losses.items():
                if k != "total_loss":
                    loss_components[k] = loss_components.get(k, 0) + v.item()
            n_batches += 1

        if self.scheduler:
            self.scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        metrics = {"train/loss": avg_loss, "train/epoch": epoch}
        for k, v in loss_components.items():
            metrics[f"train/{k}"] = v / max(n_batches, 1)
        metrics["train/lr"] = self.optimizer.param_groups[0]["lr"]

        return metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Run validation with per-task metrics."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        # Collect predictions and targets for metric computation
        task_preds = {t["name"]: [] for t in self.task_configs}
        task_targets = {t["name"]: [] for t in self.task_configs}

        for batch in self.val_loader:
            batch = self._move_batch_to_device(batch)

            output = self.model(
                audio_input_values=batch["audio_input_values"],
                attention_mask=batch["attention_mask"],
                task_type_ids=batch["task_type_ids"],
            )

            losses = self.loss_fn(
                output["task_outputs"],
                batch["clinical_features"],
            )

            total_loss += losses["total_loss"].item()
            n_batches += 1

            # Collect per-task predictions
            for task in self.task_configs:
                name = task["name"]
                input_key = task["input_key"]

                preds = output["task_outputs"][name].squeeze(-1).cpu()
                targets = batch["clinical_features"].get(input_key)
                if targets is not None:
                    targets = targets.float().squeeze(-1).cpu()
                    task_preds[name].append(preds)
                    task_targets[name].append(targets)

        avg_loss = total_loss / max(n_batches, 1)
        metrics = {"val/loss": avg_loss, "val/epoch": epoch}

        # Per-task metrics
        for task in self.task_configs:
            name = task["name"]
            task_type = task["type"]

            if not task_preds[name]:
                continue

            all_preds = torch.cat(task_preds[name]).numpy()
            all_targets = torch.cat(task_targets[name]).numpy()

            # Mask NaN targets
            valid = ~np.isnan(all_targets)
            if not valid.any():
                continue

            preds_valid = all_preds[valid]
            targets_valid = all_targets[valid]

            if task_type == "binary":
                # AUROC
                probs = 1.0 / (1.0 + np.exp(-preds_valid))  # sigmoid
                if len(set(targets_valid.astype(int))) >= 2:
                    try:
                        auroc = roc_auc_score(targets_valid, probs)
                        metrics[f"val/{name}_auroc"] = auroc
                    except ValueError:
                        pass
            elif task_type == "regression":
                # Normalize targets to match model's prediction space
                input_key = task["input_key"]
                stats = self.regression_stats.get(input_key)
                t = targets_valid
                if stats:
                    t = (t - stats["mean"]) / stats["std"]
                # RMSE in normalized space
                rmse = np.sqrt(np.mean((preds_valid - t) ** 2))
                metrics[f"val/{name}_rmse"] = rmse

        return metrics

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint and foundation model weights."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "regression_stats": self.regression_stats,
            "age_mean": self.config.get("age_mean"),
            "age_std": self.config.get("age_std"),
        }
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        path = self.checkpoint_dir / f"checkpoint_epoch{epoch:03d}.pt"
        torch.save(state, path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)

            # Also save foundation-only weights
            foundation_path = self.checkpoint_dir / "foundation_model.pt"
            torch.save(self.model.get_foundation_state_dict(), foundation_path)

            logger.info(
                "New best model saved (val_loss=%.4f). Foundation model: %s",
                val_loss, foundation_path,
            )

    def train(self, num_epochs: int, wandb_project: str | None = None):
        """Full training loop."""
        if wandb_project:
            self._init_wandb(wandb_project)

        logger.info("Starting training for %d epochs", num_epochs)
        logger.info("Device: %s, Mixed precision: %s", self.device, self.mixed_precision)

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            dt = time.time() - t0
            val_loss = val_metrics["val/loss"]

            # Build log line with available AUROC metrics
            auroc_parts = []
            for task in self.task_configs:
                name = task["name"]
                key = f"val/{name}_auroc"
                if key in val_metrics:
                    auroc_parts.append(f"{name}={val_metrics[key]:.3f}")

            logger.info(
                "Epoch %d/%d (%.1fs) - train_loss: %.4f, val_loss: %.4f%s",
                epoch, num_epochs, dt,
                train_metrics["train/loss"], val_loss,
                f" | AUROC: {', '.join(auroc_parts)}" if auroc_parts else "",
            )

            # Checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, val_loss, is_best)

            # Logging
            all_metrics = {**train_metrics, **val_metrics}
            if self._wandb:
                self._wandb.log(all_metrics, step=epoch)

            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        logger.info("Training complete. Best val_loss: %.4f", self.best_val_loss)
