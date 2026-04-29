#!/usr/bin/env python3
"""Evaluate VoiceFM on PVQD (Perceptual Voice Qualities Database).

Three evaluation tasks:
  H25: Pathological voice detection (few-shot binary classification)
  H26: GRBAS ordinal prediction (within-dataset Ridge + Spearman)
  CAPE-V continuous prediction (within-dataset Ridge + Pearson)

Compares VoiceFM (256d, 768d) against frozen HuBERT baseline.

Usage:
    python scripts/evaluate_pvqd_v3.py \
        --checkpoint checkpoints_exp_d_gsd_seed42/best_model.pt \
        --experiment exp_d_gsd_seed42 \
        --baseline
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.external_datasets import PVQDDataset
from src.models.audio_encoder import AudioEncoder
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

CAPEV_TARGETS = ["capev_severity", "capev_roughness", "capev_breathiness", "capev_strain"]
GRBAS_TARGETS = ["grbas_grade", "grbas_roughness", "grbas_breathiness", "grbas_asthenia", "grbas_strain"]
TARGET_DISPLAY = {
    "capev_severity": "severity",
    "capev_roughness": "roughness",
    "capev_breathiness": "breathiness",
    "capev_strain": "strain",
    "grbas_grade": "grade",
    "grbas_roughness": "roughness",
    "grbas_breathiness": "breathiness",
    "grbas_asthenia": "asthenia",
    "grbas_strain": "strain",
}


# ── Config / model loading ───────────────────────────────────────────────────

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
    for name in ["model", "data", "train"]:
        path = PROJECT_ROOT / "configs" / f"{name}.yaml"
        if path.exists():
            with open(path) as f:
                config[name] = yaml.safe_load(f)
    if experiment:
        exp_path = PROJECT_ROOT / "configs" / "experiments" / f"{experiment}.yaml"
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {exp_path}")
        with open(exp_path) as f:
            exp_overrides = yaml.safe_load(f)
        for section in ["model", "data", "train"]:
            if section in exp_overrides:
                config[section] = deep_merge(config[section], exp_overrides[section])
                del exp_overrides[section]
        if exp_overrides:
            config["train"] = deep_merge(config["train"], exp_overrides)
    return config


def build_audio_enc(model_cfg: dict, num_task_types: int, device: torch.device) -> AudioEncoder:
    ae_cfg = model_cfg["audio_encoder"]
    audio_encoder = AudioEncoder(
        backbone=ae_cfg["backbone"],
        freeze_layers=ae_cfg["freeze_layers"],
        projection_dim=ae_cfg["projection_dim"],
        num_task_types=num_task_types,
        spec_augment=False,
        gradient_checkpointing=False,
    )
    return audio_encoder.to(device)


# ── Collate ──────────────────────────────────────────────────────────────────

def pvqd_collate_fn(batch: list[dict]) -> dict:
    """Collate function for PVQDDataset batches."""
    return {
        "audio_input_values": torch.stack([b["audio_values"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "participant_id": [b["participant_id"] for b in batch],
        "capev": {
            k: torch.stack([b["capev"][k] for b in batch])
            for k in batch[0]["capev"]
        },
        "grbas": {
            k: torch.stack([b["grbas"][k] for b in batch])
            for k in batch[0]["grbas"]
        },
        "disease": torch.stack([b["labels"]["disease"] for b in batch]),
    }


# ── Embedding extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_voicefm_embeddings(
    audio_encoder: AudioEncoder,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Extract 256d and 768d embeddings via forward hook on AttentivePooling."""
    audio_encoder.eval()
    pooled_cache = []

    def hook_fn(module, input, output):
        pooled_cache.append(output.detach().cpu())

    handle = audio_encoder.pooling.register_forward_hook(hook_fn)

    results = {"audio_embeds_256d": [], "audio_embeds_768d": [], "participant_ids": []}

    try:
        for batch in dataloader:
            audio = batch["audio_input_values"].to(device)
            mask = batch["attention_mask"].to(device)
            task_ids = torch.zeros(audio.shape[0], dtype=torch.long, device=device)

            embeds_256d = audio_encoder(audio, mask, task_ids)
            results["audio_embeds_256d"].append(embeds_256d.cpu())
            results["participant_ids"].extend(batch["participant_id"])

        results["audio_embeds_256d"] = torch.cat(results["audio_embeds_256d"]).numpy()
        pooled_768 = torch.cat(pooled_cache)
        pooled_768 = F.normalize(pooled_768, p=2, dim=-1)
        results["audio_embeds_768d"] = pooled_768.numpy()
    finally:
        handle.remove()

    logger.info(
        "Extracted %d VoiceFM embeddings: 256d=%s, 768d=%s",
        len(results["participant_ids"]),
        results["audio_embeds_256d"].shape,
        results["audio_embeds_768d"].shape,
    )
    return results


@torch.no_grad()
def extract_hubert_baseline(
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Extract frozen HuBERT embeddings (768d)."""
    from transformers import HubertModel

    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    hubert.eval()

    results = {"audio_embeds": [], "participant_ids": []}

    for batch in dataloader:
        audio = batch["audio_input_values"].to(device)
        mask = batch["attention_mask"].to(device)

        outputs = hubert(input_values=audio, attention_mask=mask)
        hidden = outputs.last_hidden_state  # (B, T, 768)

        # Downsample mask to match HuBERT output length (CNN stride)
        frame_mask = hubert._get_feature_vector_attention_mask(hidden.shape[1], mask)
        frame_mask_f = frame_mask.unsqueeze(-1).float()
        embeds = (hidden * frame_mask_f).sum(dim=1) / frame_mask_f.sum(dim=1).clamp(min=1)
        embeds = F.normalize(embeds, p=2, dim=-1)

        results["audio_embeds"].append(embeds.cpu())
        results["participant_ids"].extend(batch["participant_id"])

    results["audio_embeds"] = torch.cat(results["audio_embeds"]).numpy()
    logger.info("Extracted %d HuBERT baseline embeddings: %s",
                len(results["participant_ids"]), results["audio_embeds"].shape)
    return results


# ── Within-dataset probing ───────────────────────────────────────────────────

def agg_by_participant(embeds: np.ndarray, pids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Average embeddings per participant. Returns (embeds_array, pid_list)."""
    unique = sorted(set(pids))
    agg_embeds = []
    for pid in unique:
        mask = [j for j, p in enumerate(pids) if p == pid]
        agg_embeds.append(embeds[mask].mean(axis=0))
    return np.array(agg_embeds), unique


def repeated_cv_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 100,
    test_frac: float = 0.2,
    seed: int = 42,
) -> dict:
    """Repeated random-split Ridge regression within a single dataset.

    For each repeat:
      - Random 80/20 train/test split
      - Fit Ridge (alpha=1.0) on train, evaluate on test
      - Record R², MAE, RMSE, Pearson r on test set

    Returns mean ± std of each metric across repeats.
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    n_test = max(int(n * test_frac), 2)

    r2_list, mae_list, rmse_list, pearson_list = [], [], [], []

    for _ in range(n_repeats):
        idx = rng.permutation(n)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        reg = Ridge(alpha=1.0)
        reg.fit(X_train_s, y_train)
        y_pred = reg.predict(X_test_s)

        r2_list.append(r2_score(y_test, y_pred))
        mae_list.append(mean_absolute_error(y_test, y_pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r_val, _ = pearsonr(y_test, y_pred)
        pearson_list.append(r_val)

    return {
        "r2_mean": float(np.mean(r2_list)),
        "r2_std": float(np.std(r2_list)),
        "mae_mean": float(np.mean(mae_list)),
        "mae_std": float(np.std(mae_list)),
        "rmse_mean": float(np.mean(rmse_list)),
        "rmse_std": float(np.std(rmse_list)),
        "pearson_mean": float(np.mean(pearson_list)),
        "pearson_std": float(np.std(pearson_list)),
        "n_total": n,
        "n_repeats": n_repeats,
    }


def few_shot_probe(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    n_trials: int = 100,
    seed: int = 42,
) -> dict:
    """Run n_trials of k-shot binary classification (k pos + k neg per trial)."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) < k + 2 or len(neg_idx) < k + 2:
        return {"mean_auroc": float("nan"), "std_auroc": 0, "n_trials": 0,
                "k": k, "n_pos": len(pos_idx), "n_neg": len(neg_idx)}

    aurocs = []
    for _ in range(n_trials):
        train_pos = rng.choice(pos_idx, k, replace=False)
        train_neg = rng.choice(neg_idx, k, replace=False)
        train_idx = np.concatenate([train_pos, train_neg])

        test_mask = np.ones(len(y), dtype=bool)
        test_mask[train_idx] = False
        test_idx = np.where(test_mask)[0]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        if len(set(y_test)) < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_train_s, y_train)
        y_prob = clf.predict_proba(X_test_s)[:, 1]

        try:
            aurocs.append(roc_auc_score(y_test, y_prob))
        except ValueError:
            continue

    if not aurocs:
        return {"mean_auroc": float("nan"), "std_auroc": 0, "n_trials": 0,
                "k": k, "n_pos": len(pos_idx), "n_neg": len(neg_idx)}

    return {
        "mean_auroc": float(np.mean(aurocs)),
        "std_auroc": float(np.std(aurocs)),
        "n_trials": len(aurocs),
        "k": k,
        "n_pos": len(pos_idx),
        "n_neg": len(neg_idx),
    }


def repeated_cv_probe_ordinal(
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 100,
    test_frac: float = 0.2,
    seed: int = 42,
) -> dict:
    """Repeated random-split Ridge regression for ordinal targets.

    Same as repeated_cv_probe but adds Spearman rank correlation (appropriate
    for ordinal GRBAS 0-3 scale).
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    n_test = max(int(n * test_frac), 2)

    r2_list, mae_list, pearson_list, spearman_list = [], [], [], []

    for _ in range(n_repeats):
        idx = rng.permutation(n)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        reg = Ridge(alpha=1.0)
        reg.fit(X_train_s, y_train)
        y_pred = reg.predict(X_test_s)

        r2_list.append(r2_score(y_test, y_pred))
        mae_list.append(mean_absolute_error(y_test, y_pred))
        r_val, _ = pearsonr(y_test, y_pred)
        pearson_list.append(r_val)
        rho, _ = spearmanr(y_test, y_pred)
        spearman_list.append(rho)

    return {
        "r2_mean": float(np.mean(r2_list)),
        "r2_std": float(np.std(r2_list)),
        "mae_mean": float(np.mean(mae_list)),
        "mae_std": float(np.std(mae_list)),
        "pearson_mean": float(np.mean(pearson_list)),
        "pearson_std": float(np.std(pearson_list)),
        "spearman_mean": float(np.mean(spearman_list)),
        "spearman_std": float(np.std(spearman_list)),
        "n_total": n,
        "n_repeats": n_repeats,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate VoiceFM on PVQD CAPE-V")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--baseline", action="store_true", help="Also run HuBERT baseline")
    parser.add_argument("--pvqd-dir", type=str, default="data/external/pvqd")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed label for output filename (e.g., 42)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    pvqd_dir = Path(args.pvqd_dir)
    if not pvqd_dir.is_absolute():
        pvqd_dir = PROJECT_ROOT / pvqd_dir

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.experiment)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # ── Load PVQD data ────────────────────────────────────────────────────
    pvqd_meta = pvqd_dir / "metadata.csv"
    pvqd_audio = pvqd_dir / "audio_16k"

    pvqd_dataset = PVQDDataset(metadata_csv=pvqd_meta, audio_dir=pvqd_audio)
    batch_size = config["train"]["training"]["batch_size"]
    pvqd_loader = torch.utils.data.DataLoader(
        pvqd_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=pvqd_collate_fn, num_workers=2,
    )
    logger.info("PVQD: %d samples", len(pvqd_dataset))

    # Build label dicts: pid -> {target: value}
    pvqd_meta_df = pd.read_csv(pvqd_meta)
    pvqd_capev_labels = {}
    pvqd_grbas_labels = {}
    pvqd_disease_labels = {}  # pid -> 0/1
    for _, row in pvqd_meta_df.iterrows():
        pid = str(row["participant_id"])
        # CAPE-V
        pvqd_capev_labels[pid] = {}
        for target in CAPEV_TARGETS:
            val = row.get(target)
            if val is not None and pd.notna(val):
                pvqd_capev_labels[pid][target] = float(val)
        # GRBAS
        pvqd_grbas_labels[pid] = {}
        for target in GRBAS_TARGETS:
            val = row.get(target)
            if val is not None and pd.notna(val):
                pvqd_grbas_labels[pid][target] = float(val)
        # Pathological status
        patho = row.get("is_pathological")
        if patho is not None and pd.notna(patho) and patho >= 0:
            pvqd_disease_labels[pid] = int(patho)

    n_patho = sum(1 for v in pvqd_disease_labels.values() if v == 1)
    n_normal = sum(1 for v in pvqd_disease_labels.values() if v == 0)
    logger.info("PVQD pathological: %d, normal: %d", n_patho, n_normal)

    # ── Load VoiceFM model ────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Infer num_task_types from checkpoint
    task_emb_key = "audio_encoder.task_embedding.weight"
    num_task_types = state_dict[task_emb_key].shape[0] if task_emb_key in state_dict else 100

    audio_encoder = build_audio_enc(config["model"], num_task_types, device)

    ae_state = {}
    prefix = "audio_encoder."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            ae_state[k[len(prefix):]] = v
    audio_encoder.load_state_dict(ae_state)
    logger.info(
        "Loaded checkpoint epoch %d (val_loss=%.4f)",
        ckpt.get("epoch", -1), ckpt.get("val_loss", float("nan")),
    )

    # ── Extract embeddings ────────────────────────────────────────────────
    logger.info("Extracting VoiceFM embeddings (PVQD)...")
    pvqd_embeds = extract_voicefm_embeddings(audio_encoder, pvqd_loader, device)
    del audio_encoder
    torch.cuda.empty_cache()

    # Aggregate per participant
    X_256d, pids_256 = agg_by_participant(
        pvqd_embeds["audio_embeds_256d"], pvqd_embeds["participant_ids"],
    )
    X_768d, pids_768 = agg_by_participant(
        pvqd_embeds["audio_embeds_768d"], pvqd_embeds["participant_ids"],
    )

    # HuBERT baseline
    X_hubert, pids_hub = None, None
    if args.baseline:
        logger.info("Extracting HuBERT baseline (PVQD)...")
        hubert_embeds = extract_hubert_baseline(pvqd_loader, device)
        X_hubert, pids_hub = agg_by_participant(
            hubert_embeds["audio_embeds"], hubert_embeds["participant_ids"],
        )

    # ── Set up methods ──────────────────────────────────────────────────
    all_metrics = {}
    methods = [
        ("voicefm_256d", X_256d, pids_256),
        ("voicefm_768d", X_768d, pids_768),
    ]
    if X_hubert is not None:
        methods.append(("hubert_768d", X_hubert, pids_hub))

    probe_seed = args.seed if args.seed is not None else 42
    method_names = [m[0] for m in methods]

    # ══════════════════════════════════════════════════════════════════════
    # H25: Pathological voice detection (few-shot probing)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=== H25: Few-shot pathological voice detection ===")
    k_values = [1, 2, 5, 10, 20]

    for method_name, X_all, pids_all in methods:
        # Build aligned X, y for disease detection
        X_list, y_list = [], []
        for i, pid in enumerate(pids_all):
            if pid in pvqd_disease_labels:
                X_list.append(X_all[i])
                y_list.append(pvqd_disease_labels[pid])
        X = np.array(X_list)
        y = np.array(y_list)

        for k in k_values:
            res = few_shot_probe(X, y, k, n_trials=args.n_repeats, seed=probe_seed)
            for metric_key, metric_val in res.items():
                all_metrics[f"h25/{method_name}/k{k}/{metric_key}"] = metric_val
            logger.info("  %s k=%d: AUROC=%.3f±%.3f (n_pos=%d, n_neg=%d)",
                        method_name, k, res["mean_auroc"], res["std_auroc"],
                        res["n_pos"], res["n_neg"])

    # Print H25 summary
    print(f"\n{'=' * 100}")
    print(f"  H25: PVQD Pathological Voice Detection ({args.n_repeats} trials per k)")
    print(f"{'=' * 100}")
    print(f"  {'k':>5s}", end="")
    for method in method_names:
        print(f"  {method:>20s}", end="")
    print()
    print(f"  {'-' * (5 + 22 * len(method_names))}")
    for k in k_values:
        print(f"  {k:>5d}", end="")
        for method in method_names:
            auroc = all_metrics.get(f"h25/{method}/k{k}/mean_auroc", float("nan"))
            std = all_metrics.get(f"h25/{method}/k{k}/std_auroc", 0)
            if np.isnan(auroc):
                print(f"  {'N/A':>20s}", end="")
            else:
                print(f"  {auroc:.3f}±{std:.3f}        ", end="")
        print()
    print()

    # ══════════════════════════════════════════════════════════════════════
    # H26: GRBAS ordinal prediction (within-PVQD Ridge + Spearman)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=== H26: GRBAS ordinal prediction ===")

    for method_name, X_all, pids_all in methods:
        logger.info("Probing %s GRBAS (within PVQD, %d repeats)...", method_name, args.n_repeats)
        for target in GRBAS_TARGETS:
            X_list, y_list = [], []
            for i, pid in enumerate(pids_all):
                if pid in pvqd_grbas_labels and target in pvqd_grbas_labels[pid]:
                    X_list.append(X_all[i])
                    y_list.append(pvqd_grbas_labels[pid][target])

            X = np.array(X_list)
            y = np.array(y_list)

            if len(y) < 10:
                logger.warning("Skipping %s/%s: too few (%d)", method_name, target, len(y))
                continue

            res = repeated_cv_probe_ordinal(X, y, n_repeats=args.n_repeats, seed=probe_seed)
            for metric_key, metric_val in res.items():
                all_metrics[f"h26/{method_name}/{target}/{metric_key}"] = metric_val

            logger.info(
                "  %s/%s: R²=%.3f, MAE=%.2f, Pearson=%.3f, Spearman=%.3f (n=%d)",
                method_name, TARGET_DISPLAY[target],
                res["r2_mean"], res["mae_mean"],
                res["pearson_mean"], res["spearman_mean"], res["n_total"],
            )

    # Print H26 summary
    print(f"\n{'=' * 100}")
    print(f"  H26: PVQD GRBAS Ordinal Prediction ({args.n_repeats} random 80/20 splits)")
    print(f"{'=' * 100}")
    header = (f"  {'Method':<16s}  {'Target':<14s}  {'Spearman ρ':>14s}  "
              f"{'Pearson r':>14s}  {'MAE':>12s}  {'N':>5s}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for method in method_names:
        for target in GRBAS_TARGETS:
            sp_m = all_metrics.get(f"h26/{method}/{target}/spearman_mean")
            if sp_m is None:
                continue
            sp_s = all_metrics[f"h26/{method}/{target}/spearman_std"]
            r_m = all_metrics[f"h26/{method}/{target}/pearson_mean"]
            r_s = all_metrics[f"h26/{method}/{target}/pearson_std"]
            mae_m = all_metrics[f"h26/{method}/{target}/mae_mean"]
            mae_s = all_metrics[f"h26/{method}/{target}/mae_std"]
            n = all_metrics[f"h26/{method}/{target}/n_total"]
            disp = TARGET_DISPLAY[target]
            print(f"  {method:<16s}  {disp:<14s}  {sp_m:>5.3f}±{sp_s:<6.3f}  "
                  f"{r_m:>5.3f}±{r_s:<6.3f}  {mae_m:>5.2f}±{mae_s:<5.2f}  {n:>5d}")
        print()

    # H26 method means
    print(f"  {'Method means (GRBAS)':}")
    print(f"  {'-' * 60}")
    for method in method_names:
        sp_vals = [all_metrics.get(f"h26/{method}/{t}/spearman_mean") for t in GRBAS_TARGETS]
        sp_vals = [v for v in sp_vals if v is not None]
        if sp_vals:
            print(f"  {method:<16s}  mean Spearman ρ={np.mean(sp_vals):.3f}")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # CAPE-V continuous prediction (within-PVQD Ridge + Pearson)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=== CAPE-V continuous prediction ===")

    for method_name, X_all, pids_all in methods:
        logger.info("Probing %s CAPE-V (within PVQD, %d repeats)...", method_name, args.n_repeats)
        for target in CAPEV_TARGETS:
            X_list, y_list = [], []
            for i, pid in enumerate(pids_all):
                if pid in pvqd_capev_labels and target in pvqd_capev_labels[pid]:
                    X_list.append(X_all[i])
                    y_list.append(pvqd_capev_labels[pid][target])

            X = np.array(X_list)
            y = np.array(y_list)

            if len(y) < 10:
                logger.warning("Skipping %s/%s: too few (%d)", method_name, target, len(y))
                continue

            res = repeated_cv_probe(X, y, n_repeats=args.n_repeats, seed=probe_seed)
            for metric_key, metric_val in res.items():
                all_metrics[f"capev/{method_name}/{target}/{metric_key}"] = metric_val

            logger.info(
                "  %s/%s: R²=%.3f±%.3f, MAE=%.1f±%.1f, r=%.3f±%.3f (n=%d)",
                method_name, TARGET_DISPLAY[target],
                res["r2_mean"], res["r2_std"],
                res["mae_mean"], res["mae_std"],
                res["pearson_mean"], res["pearson_std"],
                res["n_total"],
            )

    # Print CAPE-V summary
    print(f"\n{'=' * 100}")
    print(f"  PVQD CAPE-V: Within-Dataset Probing ({args.n_repeats} random 80/20 splits)")
    print(f"{'=' * 100}")
    header = (f"  {'Method':<16s}  {'Target':<14s}  {'R²':>14s}  {'MAE':>12s}  "
              f"{'Pearson r':>14s}  {'N':>5s}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for method in method_names:
        for target in CAPEV_TARGETS:
            r2_m = all_metrics.get(f"capev/{method}/{target}/r2_mean")
            if r2_m is None:
                continue
            r2_s = all_metrics[f"capev/{method}/{target}/r2_std"]
            mae_m = all_metrics[f"capev/{method}/{target}/mae_mean"]
            mae_s = all_metrics[f"capev/{method}/{target}/mae_std"]
            r_m = all_metrics[f"capev/{method}/{target}/pearson_mean"]
            r_s = all_metrics[f"capev/{method}/{target}/pearson_std"]
            n = all_metrics[f"capev/{method}/{target}/n_total"]
            disp = TARGET_DISPLAY[target]
            print(f"  {method:<16s}  {disp:<14s}  {r2_m:>5.3f}±{r2_s:<6.3f}  "
                  f"{mae_m:>5.1f}±{mae_s:<5.1f}  {r_m:>5.3f}±{r_s:<6.3f}  {n:>5d}")
        print()

    # CAPE-V method means
    print(f"  {'Method means (CAPE-V)':}")
    print(f"  {'-' * 60}")
    for method in method_names:
        r_vals = [all_metrics.get(f"capev/{method}/{t}/pearson_mean") for t in CAPEV_TARGETS]
        r_vals = [v for v in r_vals if v is not None]
        if r_vals:
            print(f"  {method:<16s}  mean Pearson r={np.mean(r_vals):.3f}")
    print(f"{'=' * 100}")

    # ── Save results ──────────────────────────────────────────────────────
    seed_label = args.seed if args.seed is not None else ""
    if seed_label:
        results_path = out_dir / f"pvqd_eval_seed{seed_label}.json"
    else:
        results_path = out_dir / "pvqd_eval.json"

    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
