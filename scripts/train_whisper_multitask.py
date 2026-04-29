#!/usr/bin/env python3
"""Train Whisper multi-task clinical voice foundation model.

Fine-tunes Whisper large-v2 encoder with classification + regression heads
on B2AI GSD data. Produces 1280-dim foundation embeddings.

Adapted from train_multitask.py with Whisper-specific model.

Usage:
    python scripts/train_whisper_multitask.py --experiment exp_whisper_mt_gsd_seed42
"""

import argparse
import sys
import logging
from pathlib import Path

import torch
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio_dataset import VoiceFMDataset, build_task_type_map, voicefm_collate_fn
from src.data.clinical_encoder import ClinicalFeatureProcessor
from src.data.sampler import (
    ParticipantBatchSampler,
    build_participant_strata,
    create_participant_splits,
)
from src.models.whisper_multitask_model import WhisperClinicalModel
from src.training.multitask_loss import MultiTaskLoss
from src.training.multitask_trainer import MultiTaskTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(experiment: str | None = None) -> dict:
    config = {}

    # Load Whisper multitask model config
    model_path = PROJECT_ROOT / "configs" / "whisper_multitask_model.yaml"
    with open(model_path) as f:
        config["model"] = yaml.safe_load(f)

    # Load shared data and train configs
    for name in ["data", "train"]:
        path = PROJECT_ROOT / "configs" / f"{name}.yaml"
        if path.exists():
            with open(path) as f:
                config[name] = yaml.safe_load(f)

    # Merge experiment overrides
    if experiment:
        exp_path = PROJECT_ROOT / "configs" / "experiments" / f"{experiment}.yaml"
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {exp_path}")
        with open(exp_path) as f:
            exp_overrides = yaml.safe_load(f) or {}
        logger.info("Loading experiment: %s", experiment)
        for section in ["model", "data", "train"]:
            if section in exp_overrides:
                config[section] = deep_merge(config[section], exp_overrides[section])
                del exp_overrides[section]
        if exp_overrides:
            config["train"] = deep_merge(config["train"], exp_overrides)

    return config


def main():
    parser = argparse.ArgumentParser(description="Train Whisper multi-task clinical voice model")
    parser.add_argument("--experiment", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.experiment)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load data
    data_dir = PROJECT_ROOT / "data" / "processed"
    participants = pd.read_parquet(data_dir / "participants.parquet")
    recordings = pd.read_parquet(data_dir / "recordings.parquet")

    audio_dir = PROJECT_ROOT / "data" / "audio"
    available_ids = {p.stem for p in audio_dir.glob("*.wav")}
    recordings = recordings[recordings["recording_id"].isin(available_ids)].reset_index(drop=True)
    participants = participants[participants.index.isin(recordings["record_id"].unique())]
    logger.info("Data: %d participants, %d recordings", len(participants), len(recordings))

    # Feature config + task type map
    use_gsd = config["data"].get("use_gsd", True)
    processor = ClinicalFeatureProcessor(use_gsd=use_gsd)
    feature_config = processor.get_feature_names()
    task_type_map = build_task_type_map(recordings)
    config["model"]["task_conditioning"]["num_task_types"] = len(task_type_map) + 1

    # Split participants
    split_cfg = config["data"]["splits"]
    train_ids, val_ids, test_ids = create_participant_splits(
        participants,
        train_ratio=split_cfg["train"], val_ratio=split_cfg["val"], test_ratio=split_cfg["test"],
        seed=split_cfg["seed"], stratify_col=split_cfg.get("stratify_by"),
    )
    logger.info("Splits: train=%d, val=%d, test=%d", len(train_ids), len(val_ids), len(test_ids))

    train_recordings = recordings[recordings["record_id"].isin(train_ids)]
    val_recordings = recordings[recordings["record_id"].isin(val_ids)]
    participant_strata = build_participant_strata(participants, split_cfg.get("stratify_by"))

    # Age normalization
    train_participants = participants[participants.index.isin(train_ids)]
    train_ages = train_participants["age"].replace(-1, float("nan")).dropna()
    age_mean = float(train_ages.mean())
    age_std = float(train_ages.std()) if train_ages.std() > 0 else 1.0

    # Regression normalization
    regression_stats = {}
    for task in config["model"]["tasks"]:
        if task["type"] == "regression" and task["input_key"] != "age":
            col = task["input_key"]
            if col in train_participants.columns:
                vals = train_participants[col].replace(-1, float("nan")).dropna()
                if len(vals) > 0:
                    regression_stats[col] = {"mean": float(vals.mean()), "std": float(vals.std()) or 1.0}

    # Datasets
    train_dataset = VoiceFMDataset(
        recording_manifest=train_recordings, participant_table=participants,
        audio_dir=audio_dir, task_type_map=task_type_map,
        feature_config=feature_config, age_mean=age_mean, age_std=age_std,
    )
    val_dataset = VoiceFMDataset(
        recording_manifest=val_recordings, participant_table=participants,
        audio_dir=audio_dir, task_type_map=task_type_map,
        feature_config=feature_config, age_mean=age_mean, age_std=age_std,
    )

    # Dataloaders
    train_cfg = config["train"]
    batch_size = train_cfg["training"].get("batch_size", 8)

    train_categories = (
        participant_strata.loc[train_ids] if participant_strata is not None else None
    )
    val_categories = (
        participant_strata.loc[val_ids] if participant_strata is not None else None
    )

    train_sampler = ParticipantBatchSampler(
        train_recordings, participant_categories=train_categories,
        batch_size=batch_size,
        recordings_per_participant=train_cfg["training"].get("recordings_per_participant", 1),
    )
    val_sampler = ParticipantBatchSampler(
        val_recordings, participant_categories=val_categories,
        batch_size=batch_size, drop_last=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_sampler,
        collate_fn=voicefm_collate_fn, num_workers=train_cfg["compute"]["num_workers"],
        pin_memory=train_cfg["compute"]["pin_memory"],
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_sampler=val_sampler,
        collate_fn=voicefm_collate_fn, num_workers=train_cfg["compute"]["num_workers"],
        pin_memory=train_cfg["compute"]["pin_memory"],
    )

    # Model
    model = WhisperClinicalModel(config["model"])
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %s total, %s trainable", f"{total_params:,}", f"{trainable_params:,}")

    # Loss
    loss_fn = MultiTaskLoss(config["model"]["tasks"], regression_stats=regression_stats)

    # Optimizer — backbone LR vs head LR
    opt_cfg = train_cfg["optimizer"]
    backbone_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    param_groups = [
        {"params": backbone_params, "lr": opt_cfg["lr_backbone"]},
        {"params": head_params, "lr": opt_cfg["lr_projection"]},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=opt_cfg["weight_decay"], betas=tuple(opt_cfg["betas"]))

    # Scheduler
    total_steps = train_cfg["training"]["epochs"]
    warmup_steps = int(total_steps * train_cfg["scheduler"]["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    # Checkpoint dir
    checkpoint_dir = PROJECT_ROOT / "checkpoints_whisper_multitask"
    if args.experiment:
        checkpoint_dir = PROJECT_ROOT / f"checkpoints_{args.experiment}"

    # Trainer
    trainer = MultiTaskTrainer(
        model=model, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=device,
        config={
            **train_cfg["training"],
            **train_cfg["compute"],
            "checkpoint_dir": str(checkpoint_dir),
            "task_configs": config["model"]["tasks"],
            "regression_stats": regression_stats,
            "age_mean": age_mean, "age_std": age_std,
        },
    )

    exp_name = args.experiment or "whisper_multitask"
    logger.info("=== %s ===", exp_name)
    logger.info("Backbone: Whisper large-v2 (unfreeze top %d)", config["model"]["backbone"]["unfreeze_last_n"])
    logger.info("Tasks: %s", [t["name"] for t in config["model"]["tasks"]])
    logger.info("Batch size: %d, Grad accum: %d", batch_size, train_cfg["training"]["gradient_accumulation_steps"])
    logger.info("Checkpoint dir: %s", checkpoint_dir)

    trainer.train(
        num_epochs=train_cfg["training"]["epochs"],
        wandb_project=train_cfg["logging"].get("wandb_project"),
    )


if __name__ == "__main__":
    main()
