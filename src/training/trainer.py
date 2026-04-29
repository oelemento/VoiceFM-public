"""Training loop for VoiceFM."""

import copy
import math
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

logger = logging.getLogger(__name__)


class EmbeddingQueue:
    """FIFO queue of detached embeddings (MoCo-style).

    Stores recent audio and clinical embeddings to provide additional
    negatives for contrastive loss computation.
    """

    def __init__(self, dim: int, max_size: int = 512):
        self.dim = dim
        self.max_size = max_size
        self.audio_queue = torch.zeros(0, dim)
        self.clinical_queue = torch.zeros(0, dim)

    @torch.no_grad()
    def enqueue(self, audio_embeds: torch.Tensor, clinical_embeds: torch.Tensor):
        """Add new embeddings to the queue (FIFO)."""
        self.audio_queue = torch.cat([self.audio_queue, audio_embeds.detach().cpu()])
        self.clinical_queue = torch.cat([self.clinical_queue, clinical_embeds.detach().cpu()])
        if self.audio_queue.shape[0] > self.max_size:
            self.audio_queue = self.audio_queue[-self.max_size:]
            self.clinical_queue = self.clinical_queue[-self.max_size:]

    def get(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return current queue tensors, or None if empty."""
        if self.audio_queue.shape[0] == 0:
            return None, None
        return self.audio_queue, self.clinical_queue

    def reset(self):
        """Clear the queue (call at start of each epoch)."""
        self.audio_queue = torch.zeros(0, self.dim)
        self.clinical_queue = torch.zeros(0, self.dim)


class MomentumEncoder:
    """Exponential Moving Average (EMA) model for consistent queue embeddings.

    Maintains an EMA copy of the model that produces stable embeddings for
    the memory queue, avoiding the stale embedding problem from Runs 4-6.

    Args:
        model: The model to create an EMA copy of.
        momentum: EMA decay rate. Higher = slower updates = more stable.
            Typical values: 0.996-0.999.
    """

    def __init__(self, model: nn.Module, momentum: float = 0.999) -> None:
        self.momentum = momentum
        self.ema_model = copy.deepcopy(model)
        # Freeze EMA model - it's only updated via EMA, not gradients
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA weights: θ_ema = m * θ_ema + (1 - m) * θ."""
        for ema_param, param in zip(
            self.ema_model.parameters(), model.parameters()
        ):
            ema_param.data.mul_(self.momentum).add_(
                param.data, alpha=1.0 - self.momentum
            )

    @torch.no_grad()
    def forward(self, **kwargs) -> dict:
        """Run forward pass through the EMA model."""
        self.ema_model.eval()
        return self.ema_model(**kwargs)

    def to(self, device: torch.device) -> "MomentumEncoder":
        self.ema_model = self.ema_model.to(device)
        return self


class VoiceFMTrainer:
    """Trains the VoiceFM contrastive model.

    Args:
        model: VoiceFM model
        loss_fn: VoiceFMLoss instance
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

        # Memory queue for additional contrastive negatives
        queue_size = self.config.get("queue_size", 0)
        proj_dim = self.config.get("projection_dim", 256)
        self.embed_queue = EmbeddingQueue(proj_dim, queue_size) if queue_size > 0 else None

        # Momentum encoder for consistent queue embeddings (MoCo-style)
        use_momentum = self.config.get("use_momentum_encoder", False)
        momentum_decay = self.config.get("momentum_decay", 0.999)
        if use_momentum and queue_size > 0:
            self.momentum_encoder = MomentumEncoder(model, momentum_decay).to(device)
            logger.info(f"Momentum encoder enabled (decay={momentum_decay}, queue={queue_size})")
        else:
            self.momentum_encoder = None

        self._wandb = None

    def _init_wandb(self, project: str = "voicefm"):
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

        # Site ID (for DANN)
        if "site_id" in batch:
            device_batch["site_id"] = batch["site_id"].to(self.device)

        # Clinical features are dicts of tensors
        clinical = {}
        for key, val in batch["clinical_features"].items():
            if isinstance(val, torch.Tensor):
                clinical[key] = val.to(self.device)
        device_batch["clinical_features"] = clinical

        # Pass through string metadata (not moved to device)
        device_batch["participant_id"] = batch.get("participant_id", [])

        return device_batch

    def train_epoch(self, epoch: int) -> dict:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        n_batches = 0

        self.optimizer.zero_grad()

        # Reset embedding queue at start of each epoch
        if self.embed_queue:
            self.embed_queue.reset()

        for step, batch in enumerate(self.train_loader):
            batch = self._move_batch_to_device(batch)

            with autocast(device_type=self.device.type, enabled=self.mixed_precision):
                output = self.model(
                    audio_input_values=batch["audio_input_values"],
                    attention_mask=batch["attention_mask"],
                    task_type_ids=batch["task_type_ids"],
                    clinical_features=batch["clinical_features"],
                )

                # Get targets from batch metadata
                disease_targets = batch["clinical_features"].get("disease_categories")
                age_targets = batch["clinical_features"].get("age")

                # Get queue embeddings for additional negatives
                audio_queue, clinical_queue = (None, None)
                if self.embed_queue:
                    audio_queue, clinical_queue = self.embed_queue.get()

                losses = self.loss_fn(
                    output, disease_targets, age_targets,
                    audio_queue=audio_queue,
                    clinical_queue=clinical_queue,
                    participant_ids=batch["participant_id"],
                    site_targets=batch.get("site_id"),
                )
                scaled_loss = losses["total_loss"] / self.grad_accum_steps

            self.scaler.scale(scaled_loss).backward()

            # Update queue with embeddings
            if self.embed_queue:
                if self.momentum_encoder is not None:
                    # Use momentum encoder for consistent queue embeddings
                    with torch.no_grad():
                        ema_output = self.momentum_encoder.forward(
                            audio_input_values=batch["audio_input_values"],
                            attention_mask=batch["attention_mask"],
                            task_type_ids=batch["task_type_ids"],
                            clinical_features=batch["clinical_features"],
                        )
                    self.embed_queue.enqueue(
                        ema_output["audio_embeds"], ema_output["clinical_embeds"]
                    )
                else:
                    # Fallback: detached current-model embeddings (may stale)
                    self.embed_queue.enqueue(
                        output["audio_embeds"], output["clinical_embeds"]
                    )

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Update momentum encoder after each optimizer step
                if self.momentum_encoder is not None:
                    self.momentum_encoder.update(self.model)

            total_loss += losses["total_loss"].item()
            for k, v in losses.items():
                if k != "total":
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
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        all_audio_embeds = []
        all_clinical_embeds = []

        for batch in self.val_loader:
            batch = self._move_batch_to_device(batch)

            output = self.model(
                audio_input_values=batch["audio_input_values"],
                attention_mask=batch["attention_mask"],
                task_type_ids=batch["task_type_ids"],
                clinical_features=batch["clinical_features"],
            )

            disease_targets = batch["clinical_features"].get("disease_categories")
            age_targets = batch["clinical_features"].get("age")
            losses = self.loss_fn(output, disease_targets, age_targets)

            total_loss += losses["total_loss"].item()
            n_batches += 1

            all_audio_embeds.append(output["audio_embeds"].cpu())
            all_clinical_embeds.append(output["clinical_embeds"].cpu())

        avg_loss = total_loss / max(n_batches, 1)

        # Compute retrieval metrics
        audio_embeds = torch.cat(all_audio_embeds)
        clinical_embeds = torch.cat(all_clinical_embeds)
        retrieval = self._compute_retrieval_metrics(audio_embeds, clinical_embeds)

        metrics = {"val/loss": avg_loss, "val/epoch": epoch}
        metrics.update({f"val/{k}": v for k, v in retrieval.items()})

        return metrics

    def _compute_retrieval_metrics(
        self, audio_embeds: torch.Tensor, clinical_embeds: torch.Tensor
    ) -> dict:
        """Compute cross-modal retrieval metrics."""
        # Cosine similarity
        sim = audio_embeds @ clinical_embeds.T
        n = sim.shape[0]
        targets = torch.arange(n)

        metrics = {}
        for k in [1, 5, 10]:
            # Audio -> Clinical
            topk_a2c = sim.topk(min(k, n), dim=1).indices
            recall_a2c = (topk_a2c == targets.unsqueeze(1)).any(dim=1).float().mean()

            # Clinical -> Audio
            topk_c2a = sim.T.topk(min(k, n), dim=1).indices
            recall_c2a = (topk_c2a == targets.unsqueeze(1)).any(dim=1).float().mean()

            metrics[f"recall@{k}_a2c"] = recall_a2c.item()
            metrics[f"recall@{k}_c2a"] = recall_c2a.item()

        return metrics

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        path = self.checkpoint_dir / f"checkpoint_epoch{epoch:03d}.pt"
        torch.save(state, path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            logger.info(f"New best model saved (val_loss={val_loss:.4f})")

    def _apply_temperature_schedule(self, epoch: int, num_epochs: int):
        """Clamp learned temperature to scheduled bounds each epoch."""
        temp_cfg = self.config.get("temperature_schedule")
        if not temp_cfg or not temp_cfg.get("enabled"):
            return

        temp_start = temp_cfg["temp_start"]   # 0.1
        temp_end = temp_cfg["temp_end"]       # 0.05
        warmup = temp_cfg.get("warmup_epochs", 20)

        if epoch <= warmup:
            target = temp_start
        else:
            progress = (epoch - warmup) / max(num_epochs - warmup, 1)
            target = temp_end + 0.5 * (temp_start - temp_end) * (1 + math.cos(math.pi * progress))

        # Clamp learnable log_temperature to ±20% window around target
        with torch.no_grad():
            log_min = math.log(target * 0.8)
            log_max = math.log(target * 1.2)
            self.model.log_temperature.clamp_(log_min, log_max)

    def train(self, num_epochs: int, wandb_project: str | None = None):
        """Full training loop."""
        if wandb_project:
            self._init_wandb(wandb_project)

        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}, Mixed precision: {self.mixed_precision}")

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()

            self._apply_temperature_schedule(epoch, num_epochs)

            # DANN: ramp GRL lambda from 0 to 1 over training
            if hasattr(self.model, 'site_adversarial') and self.model.site_adversarial:
                progress = epoch / num_epochs
                grl_lambda = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
                self.model.grl.set_lambda(grl_lambda)

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            dt = time.time() - t0
            val_loss = val_metrics["val/loss"]

            extra = ""
            if hasattr(self.model, 'site_adversarial') and self.model.site_adversarial:
                site_loss_val = train_metrics.get('train/site_loss', 0)
                grl_l = self.model.grl.lambda_
                extra = f", site_loss: {site_loss_val:.4f}, grl_λ: {grl_l:.3f}"

            logger.info(
                f"Epoch {epoch}/{num_epochs} ({dt:.1f}s) - "
                f"train_loss: {train_metrics['train/loss']:.4f}, "
                f"val_loss: {val_loss:.4f}, "
                f"R@1_a2c: {val_metrics.get('val/recall@1_a2c', 0):.3f}, "
                f"R@5_a2c: {val_metrics.get('val/recall@5_a2c', 0):.3f}, "
                f"temp: {self.model.temperature.item():.4f}"
                f"{extra}"
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
                logger.info(f"Early stopping at epoch {epoch}")
                break

        logger.info(f"Training complete. Best val_loss: {self.best_val_loss:.4f}")
