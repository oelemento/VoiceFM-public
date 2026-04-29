#!/usr/bin/env python3
"""Launch multi-task clinical voice foundation model training.

Usage:
    python scripts/train_multitask.py
    python scripts/train_multitask.py --experiment exp_multitask_specaug
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
from src.models.multitask_model import ClinicalVoiceModel
from src.training.multitask_loss import MultiTaskLoss
from src.training.multitask_trainer import MultiTaskTrainer
from src.data.external_datasets import (
    SVDDataset, CoswaraDataset, ExternalMultitaskDataset,
)

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
    """Load multitask model config + data/train configs, with optional experiment overrides."""
    config = {}

    # Load multitask model config
    model_path = PROJECT_ROOT / "configs" / "multitask_model.yaml"
    with open(model_path) as f:
        config["model"] = yaml.safe_load(f)

    # Load shared data and train configs
    for name in ["data", "train"]:
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
            exp_overrides = yaml.safe_load(f) or {}
        logger.info("Loading experiment: %s", experiment)
        logger.info("Overrides: %s", exp_overrides)
        for section in ["model", "data", "train"]:
            if section in exp_overrides:
                config[section] = deep_merge(config[section], exp_overrides[section])
                del exp_overrides[section]
        if exp_overrides:
            config["train"] = deep_merge(config["train"], exp_overrides)

    return config


def main():
    parser = argparse.ArgumentParser(description="Train multi-task clinical voice model")
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Experiment config name (loaded from configs/experiments/<name>.yaml)",
    )
    parser.add_argument(
        "--external-datasets", action="store_true",
        help="Include SVD + Coswara external datasets in training (Experiment 6c)",
    )
    parser.add_argument(
        "--external-data-dir", type=str, default=None,
        help="Path to external datasets dir (default: data/external)",
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
    logger.info("Using device: %s", device)

    # Load processed data
    data_dir = PROJECT_ROOT / "data" / "processed"
    participants = pd.read_parquet(data_dir / "participants.parquet")
    recordings = pd.read_parquet(data_dir / "recordings.parquet")
    logger.info("Loaded %d participants, %d recordings", len(participants), len(recordings))

    # Filter to recordings with available audio files
    audio_dir = PROJECT_ROOT / "data" / "audio"
    available_ids = {p.stem for p in audio_dir.glob("*.wav")}
    recordings = recordings[recordings["recording_id"].isin(available_ids)].reset_index(drop=True)
    participants = participants[participants.index.isin(recordings["record_id"].unique())]
    logger.info("After audio filter: %d participants, %d recordings", len(participants), len(recordings))

    # Feature config
    use_gsd = config["data"].get("use_gsd", False)
    processor = ClinicalFeatureProcessor(use_gsd=use_gsd)
    feature_config = processor.get_feature_names()

    # Task type map
    task_type_map = build_task_type_map(recordings)
    logger.info("Task types: %d", len(task_type_map))

    # Override num_task_types from actual data
    config["model"]["task_conditioning"]["num_task_types"] = len(task_type_map) + 1

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
    logger.info("Age normalization: mean=%.1f, std=%.1f", age_mean, age_std)

    # Compute regression normalization stats for questionnaire scores
    # (age is already z-scored by the dataset, so exclude it here)
    regression_stats = {}
    for task in config["model"]["tasks"]:
        if task["type"] == "regression" and task["input_key"] != "age":
            col = task["input_key"]
            if col in train_participants.columns:
                vals = train_participants[col].replace(-1, float("nan")).dropna()
                if len(vals) > 0:
                    regression_stats[col] = {
                        "mean": float(vals.mean()),
                        "std": float(vals.std()) if vals.std() > 0 else 1.0,
                    }
                    logger.info(
                        "Regression norm %s: mean=%.2f, std=%.2f (n=%d)",
                        col, regression_stats[col]["mean"], regression_stats[col]["std"], len(vals),
                    )
    logger.info("Regression normalization stats: %s", list(regression_stats.keys()))

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

    # Optionally append external datasets (Experiment 6c)
    if args.external_datasets:
        ext_dir = Path(args.external_data_dir) if args.external_data_dir else PROJECT_ROOT / "data" / "external"
        logger.info("Loading external datasets from %s", ext_dir)

        external_datasets = []

        # SVD → cat_voice
        svd_meta = ext_dir / "svd" / "metadata.csv"
        svd_audio = ext_dir / "svd" / "audio"
        if svd_meta.exists():
            svd = SVDDataset(str(svd_meta), str(svd_audio))
            svd_wrapped = ExternalMultitaskDataset(svd, target_key="cat_voice", feature_config=feature_config)
            external_datasets.append(svd_wrapped)
            logger.info("SVD: %d recordings → cat_voice", len(svd))

        # Coswara → cat_respiratory
        coswara_meta = ext_dir / "coswara" / "metadata.csv"
        coswara_audio = ext_dir / "coswara" / "audio"
        if coswara_meta.exists():
            coswara = CoswaraDataset(str(coswara_meta), str(coswara_audio))
            coswara_wrapped = ExternalMultitaskDataset(coswara, target_key="cat_respiratory", feature_config=feature_config)
            external_datasets.append(coswara_wrapped)
            logger.info("Coswara: %d recordings → cat_respiratory", len(coswara))

        if external_datasets:
            train_dataset = torch.utils.data.ConcatDataset([train_dataset] + external_datasets)
            logger.info("Combined training dataset: %d samples (B2AI + external)", len(train_dataset))

    # Samplers and Dataloaders
    train_cfg = config["train"]
    batch_size = train_cfg["training"]["batch_size"]

    if args.external_datasets and isinstance(train_dataset, torch.utils.data.ConcatDataset):
        # ConcatDataset: use standard shuffle DataLoader (ParticipantBatchSampler
        # only works with B2AI recording manifest)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=voicefm_collate_fn,
            num_workers=train_cfg["compute"]["num_workers"],
            pin_memory=train_cfg["compute"]["pin_memory"],
            drop_last=True,
        )
    else:
        # B2AI only: use participant-aware sampler
        train_sampler = ParticipantBatchSampler(
            train_recordings,
            participant_categories=train_categories,
            batch_size=batch_size,
            recordings_per_participant=train_cfg["training"].get("recordings_per_participant", 1),
            task_stratify=train_cfg["training"].get("task_stratify", True),
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=voicefm_collate_fn,
            num_workers=train_cfg["compute"]["num_workers"],
            pin_memory=train_cfg["compute"]["pin_memory"],
        )

    val_sampler = ParticipantBatchSampler(
        val_recordings,
        participant_categories=val_categories,
        batch_size=batch_size,
        task_stratify=train_cfg["training"].get("task_stratify", True),
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=voicefm_collate_fn,
        num_workers=train_cfg["compute"]["num_workers"],
        pin_memory=train_cfg["compute"]["pin_memory"],
    )

    # Model
    model_cfg = config["model"]

    # Apply spec_augment and gradient_checkpointing from train config
    spec_augment = train_cfg.get("augmentation", {}).get("spec_augment", False)
    grad_ckpt = train_cfg.get("training", {}).get("gradient_checkpointing", False)
    model_cfg["spec_augment"] = spec_augment
    model_cfg["backbone"]["gradient_checkpointing"] = grad_ckpt

    model = ClinicalVoiceModel(model_cfg)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s total, %s trainable", f"{total_params:,}", f"{trainable_params:,}")

    # Loss (with regression target normalization)
    loss_fn = MultiTaskLoss(model_cfg["tasks"], regression_stats=regression_stats)

    # Optimizer with differential learning rates
    opt_cfg = train_cfg["optimizer"]
    backbone_params = [
        p for n, p in model.named_parameters()
        if n.startswith("hubert.") and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("hubert.") and p.requires_grad
    ]
    param_groups = [
        {"params": backbone_params, "lr": opt_cfg["lr_backbone"]},
        {"params": head_params, "lr": opt_cfg["lr_projection"]},
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

    # Checkpoint dir
    checkpoint_dir = PROJECT_ROOT / "checkpoints_multitask"
    if args.experiment:
        checkpoint_dir = PROJECT_ROOT / f"checkpoints_{args.experiment}"

    # Trainer
    trainer = MultiTaskTrainer(
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
            "task_configs": model_cfg["tasks"],
            "regression_stats": regression_stats,
            "age_mean": age_mean,
            "age_std": age_std,
        },
    )

    # Log config
    exp_name = args.experiment or "multitask_baseline"
    logger.info("=== Experiment: %s ===", exp_name)
    logger.info("Backbone: %s (freeze %d layers)", model_cfg["backbone"]["name"], model_cfg["backbone"]["freeze_layers"])
    logger.info("Tasks: %s", [t["name"] for t in model_cfg["tasks"]])
    logger.info("Batch size: %d", train_cfg["training"]["batch_size"])
    logger.info("Grad accum: %d", train_cfg["training"]["gradient_accumulation_steps"])
    logger.info("SpecAugment: %s", spec_augment)
    logger.info("Gradient checkpointing: %s", grad_ckpt)
    logger.info("Checkpoint dir: %s", checkpoint_dir)
    logger.info("External datasets: %s", args.external_datasets)

    # Train
    trainer.train(
        num_epochs=train_cfg["training"]["epochs"],
        wandb_project=train_cfg["logging"].get("wandb_project"),
    )


if __name__ == "__main__":
    main()
