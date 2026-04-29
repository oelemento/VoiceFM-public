#!/usr/bin/env python3
"""H23: Fine-tune PD classifier on mPower — VoiceFM-init vs HuBERT-init.

Trains a binary Parkinson's disease classifier on mPower sustained phonation
recordings, comparing two weight initializations:
  - VoiceFM-init: Start from VoiceFM audio encoder (B2AI pre-trained)
  - HuBERT-init: Start from raw HuBERT-base (no clinical pre-training)

Same architecture, same training procedure — only initialization differs.

Also runs frozen linear probes for a no-fine-tuning baseline.

Usage:
    python scripts/finetune_pd.py \
        --init voicefm --checkpoint checkpoints_exp_d_gsd/best_model.pt \
        --metadata data/mpower/mpower_metadata.csv \
        --audio-dir data/mpower/audio \
        --seed 42 --out-dir results/mpower_pd

    python scripts/finetune_pd.py \
        --init hubert \
        --metadata data/mpower/mpower_metadata.csv \
        --audio-dir data/mpower/audio \
        --seed 42 --out-dir results/mpower_pd
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.external_datasets import MPowerDataset, external_collate_fn
from src.models import build_audio_encoder
from src.models.voicefm import VoiceFM
from src.data.clinical_encoder import ClinicalFeatureProcessor
from src.models.clinical_encoder import ClinicalEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


# ── Config loading (same pattern as evaluate_fewshot.py) ─────────────

def deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(experiment=None):
    config = {}
    for name in ["model", "data", "train"]:
        path = PROJECT_ROOT / "configs" / f"{name}.yaml"
        if path.exists():
            with open(path) as f:
                config[name] = yaml.safe_load(f)
    if experiment:
        exp_path = PROJECT_ROOT / "configs" / "experiments" / f"{experiment}.yaml"
        if exp_path.exists():
            with open(exp_path) as f:
                exp_overrides = yaml.safe_load(f) or {}
            for section in ["model", "data", "train", "loss"]:
                if section in exp_overrides:
                    config[section] = deep_merge(
                        config.get(section, {}), exp_overrides[section]
                    )
    return config


# ── Model building ───────────────────────────────────────────────────

class PDClassifier(nn.Module):
    """Audio encoder + binary classification head."""

    def __init__(self, audio_encoder: nn.Module):
        super().__init__()
        self.audio_encoder = audio_encoder
        proj_dim = audio_encoder.projection_dim
        self.classifier = nn.Linear(proj_dim, 1)

    def forward(self, audio_input_values, attention_mask, task_type_ids):
        embeds = self.audio_encoder(audio_input_values, attention_mask, task_type_ids)
        logits = self.classifier(embeds).squeeze(-1)
        return logits, embeds


def build_model_voicefm(config, checkpoint_path, device):
    """Build PDClassifier initialized from VoiceFM checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    task_emb_key = "audio_encoder.task_embedding.weight"
    num_task_types = state_dict[task_emb_key].shape[0] if task_emb_key in state_dict else 100

    model_cfg = config["model"]
    feature_config = ClinicalFeatureProcessor(use_gsd=True).get_feature_names()

    audio_encoder = build_audio_encoder(
        config=model_cfg["audio_encoder"],
        num_task_types=num_task_types,
        spec_augment=False,
        gradient_checkpointing=False,
    )
    clinical_encoder = ClinicalEncoder(
        feature_config=feature_config,
        num_layers=model_cfg["clinical_encoder"]["num_layers"],
        num_heads=model_cfg["clinical_encoder"]["num_heads"],
        hidden_dim=model_cfg["clinical_encoder"]["hidden_dim"],
        dropout=model_cfg["clinical_encoder"]["dropout"],
        projection_dim=model_cfg["clinical_encoder"]["projection_dim"],
    )
    voicefm = VoiceFM(
        audio_encoder=audio_encoder,
        clinical_encoder=clinical_encoder,
        temperature_init=model_cfg["contrastive"]["temperature_init"],
        learn_temperature=model_cfg["contrastive"]["learn_temperature"],
    )

    model_state = voicefm.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_state and v.shape == model_state[k].shape}
    voicefm.load_state_dict(filtered, strict=False)
    logger.info("Loaded VoiceFM checkpoint (epoch=%d, %d/%d keys matched)",
                ckpt.get("epoch", -1), len(filtered), len(state_dict))

    model = PDClassifier(voicefm.audio_encoder)
    return model.to(device)


def build_model_hubert(config, device):
    """Build PDClassifier initialized from raw HuBERT."""
    model_cfg = config["model"]
    audio_encoder = build_audio_encoder(
        config=model_cfg["audio_encoder"],
        num_task_types=100,
        spec_augment=False,
        gradient_checkpointing=False,
    )
    model = PDClassifier(audio_encoder)
    logger.info("Built fresh HuBERT audio encoder")
    return model.to(device)


def build_model_whisper_voicefm(checkpoint_path, device):
    """Build PDClassifier initialized from VoiceFM-Whisper (H28) checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    task_emb_key = "audio_encoder.task_embedding.weight"
    num_task_types = state_dict[task_emb_key].shape[0] if task_emb_key in state_dict else 100

    whisper_cfg = {
        "type": "whisper",
        "backbone": "openai/whisper-large-v2",
        "freeze_backbone": True,
        "unfreeze_last_n": 4,
        "projection_dim": 256,
    }

    audio_encoder = build_audio_encoder(
        config=whisper_cfg,
        num_task_types=num_task_types,
        spec_augment=False,
        gradient_checkpointing=False,
    )

    ae_state = {k.replace("audio_encoder.", "", 1): v for k, v in state_dict.items()
                if k.startswith("audio_encoder.")}
    audio_encoder.load_state_dict(ae_state)
    logger.info("Loaded VoiceFM-Whisper checkpoint (%d keys)", len(ae_state))

    model = PDClassifier(audio_encoder)
    return model.to(device)


# ── Data loading ─────────────────────────────────────────────────────

def build_split_loaders(metadata_path, audio_dir, max_train_participants, seed,
                        batch_size=16, num_workers=4, task_type_map=None):
    """Build train/val/test DataLoaders from prepared metadata CSV."""
    meta = pd.read_csv(metadata_path)

    splits = {}
    for split_name in ["train", "val", "test"]:
        split_meta = meta[meta["split"] == split_name].reset_index(drop=True)

        if split_name == "train" and max_train_participants > 0:
            rng = np.random.RandomState(seed)
            pids = split_meta["participant_id"].unique()
            pid_labels = split_meta.drop_duplicates("participant_id").set_index("participant_id")["is_pd"]

            # Stratified subsample
            pids_pd = [p for p in pids if pid_labels[p] == 1]
            pids_ctrl = [p for p in pids if pid_labels[p] == 0]
            n_pd = max(1, int(max_train_participants * len(pids_pd) / len(pids)))
            n_ctrl = max_train_participants - n_pd

            rng.shuffle(pids_pd)
            rng.shuffle(pids_ctrl)
            selected = list(pids_pd[:n_pd]) + list(pids_ctrl[:n_ctrl])
            split_meta = split_meta[split_meta["participant_id"].isin(selected)].reset_index(drop=True)
            logger.info("Subsampled train: %d participants (%d PD, %d ctrl), %d recordings",
                        len(selected), n_pd, n_ctrl, len(split_meta))

        # Write temp CSV for MPowerDataset (use PID to avoid race between array tasks)
        tmp_csv = Path(metadata_path).parent / f"_tmp_{split_name}_{os.getpid()}.csv"
        split_meta.to_csv(tmp_csv, index=False)

        ds = MPowerDataset(metadata_csv=tmp_csv, audio_dir=audio_dir,
                           task_type_map=task_type_map)
        tmp_csv.unlink(missing_ok=True)

        shuffle = split_name == "train"
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, collate_fn=external_collate_fn,
            pin_memory=True, drop_last=(split_name == "train"),
        )
        splits[split_name] = loader

        n_pd_s = split_meta["is_pd"].sum()
        logger.info("%s: %d recordings (%d PD, %d ctrl)",
                    split_name, len(ds), n_pd_s, len(ds) - n_pd_s)

    return splits


# ── Training ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device, class_weight,
                    grad_accum_steps=4, use_amp=True):
    model.train()
    total_loss = 0
    n_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        audio = batch["audio_values"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"]["disease"].float().to(device, non_blocking=True)
        task_ids = batch.get("task_type_ids", torch.zeros(audio.shape[0], dtype=torch.long)).to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp):
            logits, _ = model(audio, mask, task_ids)
            # Per-sample class weighting
            weights = torch.where(labels == 1, class_weight[1], class_weight[0])
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, weight=weights, reduction="mean"
            )
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        n_batches += 1

    # Handle remaining accumulated gradients
    if n_batches % grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, use_amp=True):
    model.eval()
    all_logits, all_labels, all_pids = [], [], []

    for batch in loader:
        audio = batch["audio_values"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"]["disease"]
        task_ids = batch.get("task_type_ids", torch.zeros(audio.shape[0], dtype=torch.long)).to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp):
            logits, _ = model(audio, mask, task_ids)

        all_logits.append(logits.cpu())
        all_labels.append(labels)
        if "participant_ids" in batch:
            all_pids.extend(batch["participant_ids"])

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))

    preds = (probs >= 0.5).astype(int)
    metrics = {
        "auroc": float(roc_auc_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "sensitivity": float(np.sum((preds == 1) & (labels == 1)) / max(np.sum(labels == 1), 1)),
        "specificity": float(np.sum((preds == 0) & (labels == 0)) / max(np.sum(labels == 0), 1)),
        "n_samples": len(labels),
        "n_positive": int(np.sum(labels == 1)),
    }

    # Participant-level metrics (aggregate predictions per participant)
    if all_pids:
        participant_metrics = compute_participant_metrics(all_pids, probs, labels)
        metrics["participant_auroc"] = participant_metrics["auroc"]
        metrics["participant_accuracy"] = participant_metrics["accuracy"]
        metrics["participant_sensitivity"] = participant_metrics["sensitivity"]
        metrics["participant_specificity"] = participant_metrics["specificity"]
        metrics["n_participants"] = participant_metrics["n_participants"]

    return metrics, probs, labels


def compute_participant_metrics(pids, probs, labels):
    """Aggregate recording-level predictions per participant, then compute metrics."""
    df = pd.DataFrame({"pid": pids, "prob": probs, "label": labels})
    agg = df.groupby("pid").agg(mean_prob=("prob", "mean"), label=("label", "first")).reset_index()

    y_true = agg["label"].values
    y_prob = agg["mean_prob"].values
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1)),
        "specificity": float(np.sum((y_pred == 0) & (y_true == 0)) / max(np.sum(y_true == 0), 1)),
        "n_participants": len(agg),
    }


# ── Frozen linear probe ──────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, loader, device, use_amp=True):
    model.eval()
    all_embeds, all_labels, all_pids = [], [], []

    for batch in loader:
        audio = batch["audio_values"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        task_ids = batch.get("task_type_ids", torch.zeros(audio.shape[0], dtype=torch.long)).to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp):
            _, embeds = model(audio, mask, task_ids)

        all_embeds.append(embeds.cpu().numpy())
        all_labels.append(batch["labels"]["disease"].numpy())
        if "participant_ids" in batch:
            all_pids.extend(batch["participant_ids"])

    return np.concatenate(all_embeds), np.concatenate(all_labels), all_pids


def frozen_linear_probe(model, train_loader, test_loader, device):
    """Train a logistic regression on frozen embeddings."""
    X_train, y_train, _ = extract_embeddings(model, train_loader, device)
    X_test, y_test, test_pids = extract_embeddings(model, test_loader, device)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", class_weight="balanced")
    clf.fit(X_train_s, y_train)
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = clf.predict(X_test_s)

    metrics = {
        "auroc": float(roc_auc_score(y_test, probs)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "sensitivity": float(np.sum((preds == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1)),
        "specificity": float(np.sum((preds == 0) & (y_test == 0)) / max(np.sum(y_test == 0), 1)),
    }

    if test_pids:
        pm = compute_participant_metrics(test_pids, probs, y_test)
        metrics["participant_auroc"] = pm["auroc"]
        metrics["n_participants"] = pm["n_participants"]

    return metrics


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune PD classifier on mPower")
    parser.add_argument("--init", choices=["voicefm", "hubert", "whisper-voicefm"], required=True,
                        help="Weight initialization: voicefm (HuBERT), hubert, or whisper-voicefm")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="VoiceFM checkpoint (required if --init voicefm)")
    parser.add_argument("--experiment", type=str, default="exp_d_gsd",
                        help="Experiment config name for model architecture")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Path to mpower_metadata.csv")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--max-train-participants", type=int, default=0,
                        help="Max training participants (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("results/mpower_pd"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--task-type-map", type=str, default=None,
                        help='JSON mapping recording_type→task_type_id, e.g. \'{"sustained":786,"countdown":2}\'')
    parser.add_argument("--skip-frozen-probe", action="store_true",
                        help="Skip frozen linear probe baseline (saves time for large datasets)")
    args = parser.parse_args()

    if args.init in ("voicefm", "whisper-voicefm") and args.checkpoint is None:
        parser.error("--checkpoint required when --init voicefm or whisper-voicefm")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # Run name
    regime = f"n{args.max_train_participants}" if args.max_train_participants > 0 else "full"
    run_name = f"{args.init}_{regime}_seed{args.seed}"
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run: %s → %s", run_name, run_dir)

    # Parse task type map (only used for voicefm — HuBERT has no pre-trained
    # task embeddings, so passing non-zero IDs causes an OOB crash)
    task_type_map = None
    if args.task_type_map and args.init in ("voicefm", "whisper-voicefm"):
        task_type_map = json.loads(args.task_type_map)
        logger.info("Task type map: %s", task_type_map)
    elif args.task_type_map and args.init not in ("voicefm", "whisper-voicefm"):
        logger.info("Ignoring --task-type-map for init=%s (no pre-trained task embeddings)", args.init)

    # ── Config ───────────────────────────────────────────────────────
    config = load_config(args.experiment)

    # ── Data ─────────────────────────────────────────────────────────
    splits = build_split_loaders(
        args.metadata, args.audio_dir,
        max_train_participants=args.max_train_participants,
        seed=args.seed, batch_size=args.batch_size,
        num_workers=args.num_workers, task_type_map=task_type_map,
    )

    # ── Model ────────────────────────────────────────────────────────
    if args.init == "voicefm":
        model = build_model_voicefm(config, args.checkpoint, device)
    elif args.init == "whisper-voicefm":
        model = build_model_whisper_voicefm(args.checkpoint, device)
    else:
        model = build_model_hubert(config, device)

    # ── Frozen linear probe (before fine-tuning) ─────────────────────
    frozen_metrics = {}
    if not args.skip_frozen_probe:
        logger.info("Running frozen linear probe...")
        frozen_metrics = frozen_linear_probe(
            model, splits["train"], splits["test"], device
        )
        logger.info("Frozen probe AUROC: %.4f", frozen_metrics["auroc"])
        if "participant_auroc" in frozen_metrics:
            logger.info("Frozen probe participant AUROC: %.4f (n=%d)",
                         frozen_metrics["participant_auroc"], frozen_metrics["n_participants"])
    else:
        logger.info("Skipping frozen linear probe")

    # ── Class weights ────────────────────────────────────────────────
    meta = pd.read_csv(args.metadata)
    train_meta = meta[meta["split"] == "train"]
    n_pos = train_meta["is_pd"].sum()
    n_neg = len(train_meta) - n_pos
    w_pos = n_neg / max(n_pos, 1)
    w_neg = 1.0
    class_weight = torch.tensor([w_neg, w_pos], device=device)
    logger.info("Class weights: ctrl=%.2f, PD=%.2f", w_neg, w_pos)

    # ── Optimizer (differential LR) ──────────────────────────────────
    backbone_prefixes = ("audio_encoder.hubert.", "audio_encoder.encoder.")
    backbone_params = [p for n, p in model.named_parameters()
                       if any(n.startswith(bp) for bp in backbone_prefixes) and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if not any(n.startswith(bp) for bp in backbone_prefixes) and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=0.01)

    total_steps = args.epochs * len(splits["train"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    # Disable AMP for Whisper — FP16 gradient issues with large-v2
    use_amp = device.type == "cuda" and args.init != "whisper-voicefm"
    scaler = GradScaler(enabled=use_amp)

    # ── Training loop ────────────────────────────────────────────────
    best_val_auroc = 0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, splits["train"], optimizer, scaler, device,
            class_weight, grad_accum_steps=args.grad_accum, use_amp=use_amp,
        )
        scheduler.step()

        val_metrics, _, _ = evaluate(model, splits["val"], device, use_amp=use_amp)
        elapsed = time.time() - t0

        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auroc": val_metrics["auroc"],
            "val_f1": val_metrics["f1"],
        }
        if "participant_auroc" in val_metrics:
            history_entry["val_participant_auroc"] = val_metrics["participant_auroc"]
        history.append(history_entry)

        p_auroc_str = ""
        if "participant_auroc" in val_metrics:
            p_auroc_str = f" p_auroc={val_metrics['participant_auroc']:.4f}"
        logger.info(
            "Epoch %d/%d: loss=%.4f val_auroc=%.4f%s val_f1=%.4f [%.0fs]",
            epoch, args.epochs, train_loss,
            val_metrics["auroc"], p_auroc_str, val_metrics["f1"], elapsed,
        )

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_auroc": best_val_auroc,
            }, run_dir / "best_model.pt")
            logger.info("  → New best val AUROC: %.4f", best_val_auroc)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # ── Final evaluation on test set ─────────────────────────────────
    best_ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics, _, _ = evaluate(model, splits["test"], device, use_amp=use_amp)
    logger.info("Test AUROC: %.4f, F1: %.4f, Sens: %.4f, Spec: %.4f",
                test_metrics["auroc"], test_metrics["f1"],
                test_metrics["sensitivity"], test_metrics["specificity"])
    if "participant_auroc" in test_metrics:
        logger.info("Test participant-level AUROC: %.4f (n=%d)",
                     test_metrics["participant_auroc"], test_metrics["n_participants"])

    # ── Save results ─────────────────────────────────────────────────
    results = {
        "init": args.init,
        "regime": regime,
        "seed": args.seed,
        "max_train_participants": args.max_train_participants,
        "best_epoch": int(best_ckpt["epoch"]),
        "best_val_auroc": best_val_auroc,
        "test": test_metrics,
        "frozen_probe": frozen_metrics,
        "history": history,
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", run_dir / "results.json")

    # ── Print summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"{'='*60}")
    print(f"  Init:       {args.init}")
    print(f"  Regime:     {regime}")
    print(f"  Best epoch: {best_ckpt['epoch']}")
    print(f"  Val AUROC:  {best_val_auroc:.4f}")
    print(f"  Test AUROC (recording): {test_metrics['auroc']:.4f}")
    print(f"  Test F1:    {test_metrics['f1']:.4f}")
    print(f"  Test Sens:  {test_metrics['sensitivity']:.4f}")
    print(f"  Test Spec:  {test_metrics['specificity']:.4f}")
    if "participant_auroc" in test_metrics:
        print(f"  Test AUROC (participant): {test_metrics['participant_auroc']:.4f}  (n={test_metrics['n_participants']})")
    print(f"  Frozen AUROC (recording): {frozen_metrics['auroc']:.4f}")
    if "participant_auroc" in frozen_metrics:
        print(f"  Frozen AUROC (participant): {frozen_metrics['participant_auroc']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
