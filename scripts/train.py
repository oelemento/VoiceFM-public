#!/usr/bin/env python3
"""Launch VoiceFM training.

Usage:
    python scripts/train.py                          # baseline (train.yaml defaults)
    python scripts/train.py --experiment exp_a_large_batch  # experiment A
"""

import argparse
import sys
import logging
from pathlib import Path

import torch
import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio_dataset import VoiceFMDataset, build_task_type_map, voicefm_collate_fn
from src.data.clinical_encoder import ClinicalFeatureProcessor
from src.data.sampler import (
    ParticipantBatchSampler,
    build_participant_strata,
    create_participant_splits,
)
from src.models import build_audio_encoder
from src.models.clinical_encoder import ClinicalEncoder
from src.models.voicefm import VoiceFM
from src.training.losses import VoiceFMLoss
from src.training.trainer import VoiceFMTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(experiment: str | None = None) -> dict:
    """Load and merge config files, with optional experiment overrides."""
    config = {}
    for name in ["model", "data", "train"]:
        path = PROJECT_ROOT / "configs" / f"{name}.yaml"
        if path.exists():
            with open(path) as f:
                config[name] = yaml.safe_load(f)

    # Merge experiment-specific overrides
    if experiment:
        exp_path = PROJECT_ROOT / "configs" / "experiments" / f"{experiment}.yaml"
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {exp_path}")
        with open(exp_path) as f:
            exp_overrides = yaml.safe_load(f)
        logger.info(f"Loading experiment: {experiment}")
        logger.info(f"Overrides: {exp_overrides}")
        # Merge each top-level section into the appropriate config
        for section in ["model", "data", "train"]:
            if section in exp_overrides:
                config[section] = deep_merge(config[section], exp_overrides[section])
                del exp_overrides[section]
        # Remaining keys merge into train config (backward compat)
        if exp_overrides:
            config["train"] = deep_merge(config["train"], exp_overrides)

    return config


def main():
    parser = argparse.ArgumentParser(description="Train VoiceFM")
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Experiment config name (e.g. exp_a_large_batch). "
             "Loaded from configs/experiments/<name>.yaml",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data directory (default: data/processed). "
             "Used to route v3 retraining to data/processed_v3 without touching v2.3.0 artifacts.",
    )
    args = parser.parse_args()

    config = load_config(args.experiment)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load processed data
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "processed"
    participants = pd.read_parquet(data_dir / "participants.parquet")
    recordings = pd.read_parquet(data_dir / "recordings.parquet")

    logger.info(f"Loaded {len(participants)} participants, {len(recordings)} recordings")

    # Filter to recordings with available audio files
    audio_dir = PROJECT_ROOT / "data" / "audio"
    available_ids = {p.stem for p in audio_dir.glob("*.wav")}
    recordings = recordings[recordings["recording_id"].isin(available_ids)].reset_index(drop=True)
    # Also filter participants to those with recordings
    participants = participants[participants.index.isin(recordings["record_id"].unique())]
    logger.info(f"After audio filter: {len(participants)} participants, {len(recordings)} recordings")

    # Feature config
    use_gsd = config["data"].get("use_gsd", False)
    processor = ClinicalFeatureProcessor(use_gsd=use_gsd)
    feature_config = processor.get_feature_names()
    exclude = config["data"].get("exclude_features", [])
    if exclude:
        feature_config["binary"] = [f for f in feature_config["binary"] if f not in exclude]
        feature_config["continuous"] = [f for f in feature_config["continuous"] if f not in exclude]
        feature_config["categorical"] = {k: v for k, v in feature_config["categorical"].items() if k not in exclude}
        logger.info(f"Excluded {len(exclude)} features, remaining: {len(feature_config['binary'])} binary, "
                     f"{len(feature_config['continuous'])} continuous, {len(feature_config['categorical'])} categorical")

    # Task type map
    task_type_map = build_task_type_map(recordings)
    logger.info(f"Task types: {len(task_type_map)}")

    # Split participants
    split_cfg = config["data"]["splits"]
    stratify_col = split_cfg.get("stratify_by")
    train_ids, val_ids, test_ids = create_participant_splits(
        participants,
        train_ratio=split_cfg["train"],
        val_ratio=split_cfg["val"],
        test_ratio=split_cfg["test"],
        seed=split_cfg["seed"],
        stratify_col=stratify_col,
    )
    logger.info(
        "Splits: train=%d, val=%d, test=%d (stratify_by=%s)",
        len(train_ids), len(val_ids), len(test_ids), stratify_col,
    )

    # Filter recordings by split
    train_recordings = recordings[recordings["record_id"].isin(train_ids)]
    val_recordings = recordings[recordings["record_id"].isin(val_ids)]
    participant_strata = build_participant_strata(participants, stratify_col)
    use_category_stratify = config["train"]["training"].get("category_stratify", True)
    train_categories = (
        participant_strata.loc[train_ids] if use_category_stratify and participant_strata is not None else None
    )
    val_categories = (
        participant_strata.loc[val_ids] if use_category_stratify and participant_strata is not None else None
    )

    # Compute age normalization stats from training participants only
    train_participants = participants[participants.index.isin(train_ids)]
    train_ages = train_participants["age"].replace(-1, float("nan")).dropna()
    age_mean = float(train_ages.mean())
    age_std = float(train_ages.std()) if train_ages.std() > 0 else 1.0
    logger.info(f"Age normalization: mean={age_mean:.1f}, std={age_std:.1f}")

    # Datasets
    train_dataset = VoiceFMDataset(
        recording_manifest=train_recordings,
        participant_table=participants,
        audio_dir=audio_dir,
        task_type_map=task_type_map,
        feature_config=feature_config,
        age_mean=age_mean,
        age_std=age_std,
    )
    val_dataset = VoiceFMDataset(
        recording_manifest=val_recordings,
        participant_table=participants,
        audio_dir=audio_dir,
        task_type_map=task_type_map,
        feature_config=feature_config,
        age_mean=age_mean,
        age_std=age_std,
    )

    # Samplers
    train_sampler = ParticipantBatchSampler(
        train_recordings,
        participant_categories=train_categories,
        batch_size=config["train"]["training"]["batch_size"],
        recordings_per_participant=config["train"]["training"].get("recordings_per_participant", 1),
        task_stratify=config["train"]["training"].get("task_stratify", True),
    )
    val_sampler = ParticipantBatchSampler(
        val_recordings,
        participant_categories=val_categories,
        batch_size=config["train"]["training"]["batch_size"],
        task_stratify=config["train"]["training"].get("task_stratify", True),
        drop_last=False,
    )

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=voicefm_collate_fn,
        num_workers=config["train"]["compute"]["num_workers"],
        pin_memory=config["train"]["compute"]["pin_memory"],
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=voicefm_collate_fn,
        num_workers=config["train"]["compute"]["num_workers"],
        pin_memory=config["train"]["compute"]["pin_memory"],
    )

    # Model
    model_cfg = config["model"]
    train_cfg = config["train"]

    # Check for augmentation and checkpointing flags
    spec_augment = train_cfg.get("augmentation", {}).get("spec_augment", False)
    grad_ckpt = train_cfg.get("training", {}).get("gradient_checkpointing", False)

    audio_encoder = build_audio_encoder(
        config=model_cfg["audio_encoder"],
        num_task_types=len(task_type_map) + 1,
        spec_augment=spec_augment,
        gradient_checkpointing=grad_ckpt,
    )
    clinical_encoder = ClinicalEncoder(
        feature_config=feature_config,
        num_layers=model_cfg["clinical_encoder"]["num_layers"],
        num_heads=model_cfg["clinical_encoder"]["num_heads"],
        hidden_dim=model_cfg["clinical_encoder"]["hidden_dim"],
        dropout=model_cfg["clinical_encoder"]["dropout"],
        projection_dim=model_cfg["clinical_encoder"]["projection_dim"],
    )
    model = VoiceFM(
        audio_encoder=audio_encoder,
        clinical_encoder=clinical_encoder,
        temperature_init=model_cfg["contrastive"]["temperature_init"],
        learn_temperature=model_cfg["contrastive"]["learn_temperature"],
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss (with optional hard negative mining)
    loss_cfg = train_cfg.get("loss", {})
    loss_fn = VoiceFMLoss(
        disease_weight=model_cfg["auxiliary"]["disease_category_weight"],
        age_weight=model_cfg["auxiliary"]["age_regression_weight"],
        hard_negative_mining=loss_cfg.get("hard_negative_mining", False),
        hard_negative_beta=loss_cfg.get("hard_negative_beta", 1.0),
        contrastive_loss_type=loss_cfg.get("contrastive_loss_type", "infonce"),
    )

    # Optimizer with differential learning rates
    opt_cfg = train_cfg["optimizer"]
    backbone_prefixes = ("audio_encoder.hubert.", "audio_encoder.hear.", "audio_encoder.encoder.")
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                     if n.startswith(backbone_prefixes) and p.requires_grad],
         "lr": opt_cfg["lr_backbone"]},
        {"params": [p for n, p in model.named_parameters()
                     if not n.startswith(backbone_prefixes) and p.requires_grad],
         "lr": opt_cfg["lr_projection"]},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]),
    )

    # Scheduler
    total_steps = train_cfg["training"]["epochs"]
    warmup_steps = int(total_steps * train_cfg["scheduler"]["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps,
    )

    # Checkpoint dir — experiment-specific if running an experiment
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    if args.experiment:
        checkpoint_dir = PROJECT_ROOT / f"checkpoints_{args.experiment}"

    # Trainer
    trainer = VoiceFMTrainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config={
            **train_cfg["training"],
            **train_cfg["compute"],
            "checkpoint_dir": str(checkpoint_dir),
            "projection_dim": model_cfg["audio_encoder"]["projection_dim"],
            "temperature_schedule": train_cfg.get("temperature_schedule"),
        },
    )

    # Log experiment config
    exp_name = args.experiment or "baseline"
    encoder_type = model_cfg["audio_encoder"].get("type", "hubert")
    logger.info(f"=== Experiment: {exp_name} ===")
    logger.info(f"Audio encoder: {encoder_type} ({model_cfg['audio_encoder']['backbone']})")
    logger.info(f"Batch size: {train_cfg['training']['batch_size']}")
    logger.info(f"Grad accum: {train_cfg['training']['gradient_accumulation_steps']}")
    logger.info(f"SpecAugment: {spec_augment}")
    logger.info(f"Queue size: {train_cfg['training'].get('queue_size', 0)}")
    logger.info(f"Momentum encoder: {train_cfg['training'].get('use_momentum_encoder', False)}")
    logger.info(f"Contrastive loss: {loss_cfg.get('contrastive_loss_type', 'infonce')}")
    logger.info(f"Hard negatives: {loss_cfg.get('hard_negative_mining', False)}")
    logger.info(f"Gradient checkpointing: {grad_ckpt}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")

    # Train
    trainer.train(
        num_epochs=train_cfg["training"]["epochs"],
        wandb_project=train_cfg["logging"].get("wandb_project"),
    )


if __name__ == "__main__":
    main()
