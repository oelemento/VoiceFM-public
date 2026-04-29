#!/usr/bin/env python3
"""Evaluate whether VoiceFM embeddings encode classic acoustic voice measures.

Extracts embeddings from prolonged vowel recordings, pairs them with
Parselmouth-derived acoustic features (jitter, shimmer, HNR, F0, formants,
CPPS), and tests the relationship via:
  1. PCA + Spearman correlation heatmap (embedding PCs vs acoustic features)
  2. Ridge regression R² (predict each acoustic feature from embeddings)
  3. CCA (canonical correlation analysis)

Compares VoiceFM (256d, 768d) vs frozen HuBERT (768d).

Usage:
    python3.11 scripts/evaluate_acoustic_grounding_v3.py \
        --checkpoint checkpoints_exp_d_hard_negatives/best_model.pt \
        --acoustic-features data/processed/acoustic_features.parquet \
        --baseline --out-dir figures/acoustic_grounding
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
import yaml
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio_dataset import VoiceFMDataset, build_task_type_map, voicefm_collate_fn
from src.data.clinical_encoder import ClinicalFeatureProcessor
from src.data.sampler import (
    ParticipantBatchSampler,
    build_participant_strata,
    create_participant_splits,
)
from src.models.audio_encoder import AudioEncoder
from src.training.evaluate import extract_hubert_baseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

ACOUSTIC_FEATURES = [
    "f0_mean", "f0_sd", "f0_min", "f0_max", "f0_range",
    "jitter_local", "jitter_rap",
    "shimmer_local", "shimmer_apq3",
    "hnr",
    "f1_mean", "f2_mean", "f3_mean",
    "cpps",
]

FEATURE_LABELS = {
    "f0_mean": "F0 mean (Hz)",
    "f0_sd": "F0 SD (Hz)",
    "f0_min": "F0 min (Hz)",
    "f0_max": "F0 max (Hz)",
    "f0_range": "F0 range (Hz)",
    "jitter_local": "Jitter (local)",
    "jitter_rap": "Jitter (RAP)",
    "shimmer_local": "Shimmer (local)",
    "shimmer_apq3": "Shimmer (APQ3)",
    "hnr": "HNR (dB)",
    "f1_mean": "F1 mean (Hz)",
    "f2_mean": "F2 mean (Hz)",
    "f3_mean": "F3 mean (Hz)",
    "cpps": "CPPS (dB)",
}


# ── Config / model loading (same as evaluate_capev.py) ────────────────────────

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


def build_audio_encoder(model_cfg: dict, num_task_types: int, device: torch.device) -> AudioEncoder:
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


# ── Embedding extraction (same pattern as evaluate_capev.py) ──────────────────

@torch.no_grad()
def extract_embeddings_with_hook(
    audio_encoder: AudioEncoder,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    audio_encoder.eval()
    pooled_cache = []

    def hook_fn(module, input, output):
        pooled_cache.append(output.detach().cpu())

    handle = audio_encoder.pooling.register_forward_hook(hook_fn)

    results = {
        "audio_embeds_256d": [],
        "audio_embeds_768d": [],
        "participant_ids": [],
        "recording_ids": [],
    }

    try:
        for batch in dataloader:
            audio = batch["audio_input_values"].to(device)
            mask = batch["attention_mask"].to(device)
            task_ids = batch["task_type_id"].to(device)
            embeds_256d = audio_encoder(audio, mask, task_ids)
            results["audio_embeds_256d"].append(embeds_256d.cpu())
            results["participant_ids"].extend(batch["participant_id"])
            results["recording_ids"].extend(batch["recording_id"])

        results["audio_embeds_256d"] = torch.cat(results["audio_embeds_256d"]).numpy()
        pooled_768 = torch.cat(pooled_cache)
        pooled_768 = F.normalize(pooled_768, p=2, dim=-1)
        results["audio_embeds_768d"] = pooled_768.numpy()
    finally:
        handle.remove()

    logger.info(
        "Extracted %d embeddings: 256d=%s, 768d=%s",
        len(results["participant_ids"]),
        results["audio_embeds_256d"].shape,
        results["audio_embeds_768d"].shape,
    )
    return results


# ── Per-participant aggregation ───────────────────────────────────────────────

def _agg(embeds: np.ndarray, pids: list[str]) -> dict[str, np.ndarray]:
    unique = sorted(set(pids))
    agg = {}
    for pid in unique:
        mask = [j for j, p in enumerate(pids) if p == pid]
        agg[pid] = embeds[mask].mean(axis=0)
    return agg


def _agg_acoustic(acoustic_df: pd.DataFrame) -> pd.DataFrame:
    """Mean-pool acoustic features per participant."""
    return acoustic_df.groupby("record_id")[ACOUSTIC_FEATURES].mean()


# ── Analysis functions ────────────────────────────────────────────────────────

def pca_spearman_analysis(
    embeddings: np.ndarray,
    acoustic_features: np.ndarray,
    feature_names: list[str],
    n_components: int = 20,
) -> dict:
    """PCA on embeddings, Spearman correlation with acoustic features."""
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(embeddings)

    n_features = len(feature_names)
    rho_matrix = np.full((n_features, n_components), np.nan)
    pval_matrix = np.full((n_features, n_components), np.nan)

    for i in range(n_features):
        feat_vals = acoustic_features[:, i]
        valid = ~np.isnan(feat_vals)
        if valid.sum() < 10:
            continue
        for j in range(n_components):
            rho, pval = stats.spearmanr(pcs[valid, j], feat_vals[valid])
            rho_matrix[i, j] = rho
            pval_matrix[i, j] = pval

    return {
        "rho_matrix": rho_matrix,
        "pval_matrix": pval_matrix,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "pca_components": pcs,
    }


def ridge_cv_analysis(
    embeddings: np.ndarray,
    acoustic_features: np.ndarray,
    feature_names: list[str],
    method_name: str,
    n_folds: int = 5,
) -> dict:
    """5-fold CV Ridge regression: predict each acoustic feature from embeddings."""
    from sklearn.pipeline import Pipeline
    results = {}

    for i, feat in enumerate(feature_names):
        y = acoustic_features[:, i]
        valid = ~np.isnan(y)
        if valid.sum() < 20:
            logger.warning("Skipping %s/%s: only %d valid samples", method_name, feat, valid.sum())
            continue

        X_v = embeddings[valid]
        y_v = y[valid]

        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        y_pred = cross_val_predict(pipe, X_v, y_v, cv=n_folds)

        r2 = float(1 - np.sum((y_v - y_pred) ** 2) / np.sum((y_v - y_v.mean()) ** 2))
        rho, _ = stats.spearmanr(y_v, y_pred)
        mae = float(np.mean(np.abs(y_v - y_pred)))

        results[f"{method_name}/{feat}/r2"] = r2
        results[f"{method_name}/{feat}/spearman"] = float(rho)
        results[f"{method_name}/{feat}/mae"] = mae
        results[f"{method_name}/{feat}/n"] = int(valid.sum())
        results[f"{method_name}/{feat}/y_true"] = y_v.tolist()
        results[f"{method_name}/{feat}/y_pred"] = y_pred.tolist()

    return results


def cca_analysis(
    embeddings: np.ndarray,
    acoustic_features: np.ndarray,
    n_components: int = 5,
) -> list[float]:
    """CCA between embeddings and acoustic features. Returns canonical correlations."""
    # Drop rows with any NaN in acoustic features
    valid = ~np.any(np.isnan(acoustic_features), axis=1)
    if valid.sum() < 30:
        logger.warning("CCA: too few valid samples (%d)", valid.sum())
        return []

    X = StandardScaler().fit_transform(embeddings[valid])
    Y = StandardScaler().fit_transform(acoustic_features[valid])

    n_comp = min(n_components, Y.shape[1], X.shape[1])
    cca = CCA(n_components=n_comp)
    X_c, Y_c = cca.fit_transform(X, Y)

    correlations = []
    for i in range(n_comp):
        r, _ = stats.pearsonr(X_c[:, i], Y_c[:, i])
        correlations.append(float(r))

    return correlations


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    rho_matrix: np.ndarray,
    pval_matrix: np.ndarray,
    feature_names: list[str],
    explained_var: list[float],
    out_path: Path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_features, n_pcs = rho_matrix.shape
    # Bonferroni correction
    n_tests = n_features * n_pcs
    alpha = 0.05

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(rho_matrix, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")

    # Add significance stars
    for i in range(n_features):
        for j in range(n_pcs):
            p = pval_matrix[i, j]
            if not np.isnan(p) and p < alpha / n_tests:
                ax.text(j, i, "*", ha="center", va="center", fontsize=8, fontweight="bold")

    ax.set_xticks(range(n_pcs))
    pc_labels = [f"PC{j+1}\n({explained_var[j]*100:.1f}%)" for j in range(n_pcs)]
    ax.set_xticklabels(pc_labels, fontsize=7)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in feature_names], fontsize=9)
    ax.set_xlabel("Embedding PCA component", fontsize=11)
    ax.set_ylabel("Acoustic feature", fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman ρ", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)


def plot_ridge_r2_comparison(all_metrics: dict, methods: list[str], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = [f for f in ACOUSTIC_FEATURES if any(
        f"{m}/{f}/r2" in all_metrics for m in methods
    )]
    method_labels = {
        "voicefm_256d": "VoiceFM (256d)",
        "voicefm_768d": "VoiceFM (768d)",
        "hubert_768d": "HuBERT (768d)",
    }
    colors = {
        "voicefm_256d": "#2196F3",
        "voicefm_768d": "#1565C0",
        "hubert_768d": "#FF9800",
    }

    n_feat = len(features)
    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    x = np.arange(n_feat)

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, method in enumerate(methods):
        r2_vals = [all_metrics.get(f"{method}/{f}/r2", np.nan) for f in features]
        offset = (i - (n_methods - 1) / 2) * bar_width
        ax.bar(
            x + offset, r2_vals, bar_width,
            label=method_labels.get(method, method),
            color=colors.get(method, "#999999"),
            alpha=0.85,
        )

    ax.axhline(0.0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [FEATURE_LABELS.get(f, f) for f in features],
        fontsize=8, rotation=45, ha="right",
    )
    ax.set_ylabel("R² (5-fold CV)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0] - 0.05))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)


def plot_scatter_top_features(
    all_metrics: dict,
    best_method: str,
    out_path: Path,
    n_top: int = 6,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Find top features by R²
    feat_r2 = []
    for f in ACOUSTIC_FEATURES:
        r2 = all_metrics.get(f"{best_method}/{f}/r2")
        if r2 is not None and not np.isnan(r2):
            feat_r2.append((f, r2))
    feat_r2.sort(key=lambda x: x[1], reverse=True)
    top_feats = feat_r2[:n_top]

    nrows = 2
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9))
    axes = axes.flatten()

    for idx, (feat, r2) in enumerate(top_feats):
        ax = axes[idx]
        y_true = np.array(all_metrics[f"{best_method}/{feat}/y_true"])
        y_pred = np.array(all_metrics[f"{best_method}/{feat}/y_pred"])
        rho = all_metrics.get(f"{best_method}/{feat}/spearman", np.nan)

        ax.scatter(y_true, y_pred, alpha=0.4, s=15, edgecolors="none", c="#1565C0")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, "r--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Actual", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.set_title(f"{FEATURE_LABELS.get(feat, feat)}\nR²={r2:.3f}, ρ={rho:.3f}", fontsize=10)

    for idx in range(len(top_feats), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"Top {n_top} Acoustic Features — {best_method}", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)


def plot_pca_acoustic_coloring(
    pca_components: np.ndarray,
    acoustic_features: np.ndarray,
    feature_names: list[str],
    out_path: Path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color_feats = ["hnr", "jitter_local", "f0_mean"]
    color_feats = [f for f in color_feats if f in feature_names]

    fig, axes = plt.subplots(1, len(color_feats), figsize=(5 * len(color_feats), 4.5))
    if len(color_feats) == 1:
        axes = [axes]

    for ax, feat in zip(axes, color_feats):
        feat_idx = feature_names.index(feat)
        vals = acoustic_features[:, feat_idx]
        valid = ~np.isnan(vals)

        sc = ax.scatter(
            pca_components[valid, 0], pca_components[valid, 1],
            c=vals[valid], cmap="viridis", alpha=0.6, s=15, edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate acoustic feature grounding of embeddings")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--acoustic-features", type=str, required=True)
    parser.add_argument("--baseline", action="store_true", help="Also run HuBERT baseline")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="figures/acoustic_grounding")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--device", type=str, default=None, help="Force device: cpu, mps, cuda")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    acoustic_path = Path(args.acoustic_features)
    if not acoustic_path.is_absolute():
        acoustic_path = PROJECT_ROOT / args.acoustic_features
    if not acoustic_path.exists():
        logger.error("Acoustic features not found: %s", args.acoustic_features)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.experiment)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # Load data
    data_dir = PROJECT_ROOT / "data" / "processed_v3"
    participants = pd.read_parquet(data_dir / "participants.parquet")
    recordings = pd.read_parquet(data_dir / "recordings.parquet")
    acoustic_df = pd.read_parquet(acoustic_path)
    logger.info(
        "Loaded %d participants, %d recordings, %d acoustic features",
        len(participants), len(recordings), len(acoustic_df),
    )

    # Filter to prolonged vowel recordings only
    vowel_mask = recordings["recording_name"].str.lower() == "prolonged vowel"
    vowel_recordings = recordings[vowel_mask].reset_index(drop=True)

    # Filter to available audio
    audio_dir = PROJECT_ROOT / "data" / "audio"
    available_ids = {p.stem for p in audio_dir.glob("*.wav")}
    vowel_recordings = vowel_recordings[
        vowel_recordings["recording_id"].isin(available_ids)
    ].reset_index(drop=True)
    participants = participants[
        participants.index.isin(vowel_recordings["record_id"].unique())
    ]
    logger.info(
        "Prolonged vowel: %d recordings, %d participants",
        len(vowel_recordings), len(participants),
    )

    # Feature config and task types (need full recordings for task type map)
    processor = ClinicalFeatureProcessor()
    feature_config = processor.get_feature_names()
    task_type_map = build_task_type_map(recordings)  # Full recordings for complete map

    # Splits (same as training — use all participants, not just vowel subset)
    all_participants = pd.read_parquet(data_dir / "participants.parquet")
    split_cfg = config["data"]["splits"]
    stratify_col = split_cfg.get("stratify_by")
    train_ids, val_ids, test_ids = create_participant_splits(
        all_participants,
        train_ratio=split_cfg["train"],
        val_ratio=split_cfg["val"],
        test_ratio=split_cfg["test"],
        seed=split_cfg["seed"],
        stratify_col=stratify_col,
    )
    logger.info("Splits: train=%d, val=%d, test=%d", len(train_ids), len(val_ids), len(test_ids))

    # Use train+val for fitting, test for evaluation
    fit_ids = list(train_ids) + list(val_ids)
    fit_recordings = vowel_recordings[vowel_recordings["record_id"].isin(fit_ids)]
    test_recordings_df = vowel_recordings[vowel_recordings["record_id"].isin(test_ids)]
    logger.info(
        "Vowel: fit=%d recordings, test=%d recordings",
        len(fit_recordings), len(test_recordings_df),
    )

    # Build datasets for both splits
    participant_strata = build_participant_strata(all_participants, stratify_col)
    use_cat_strat = config["train"]["training"].get("category_stratify", True)

    train_participants = all_participants[all_participants.index.isin(train_ids)]
    train_ages = train_participants["age"].replace(-1, float("nan")).dropna()
    age_mean = float(train_ages.mean())
    age_std = float(train_ages.std()) if train_ages.std() > 0 else 1.0

    batch_size = args.batch_size if args.batch_size is not None else config["train"]["training"]["batch_size"]
    num_workers = min(config["train"]["compute"]["num_workers"], 2)

    # Build datasets and loaders for both splits
    datasets_and_loaders = {}
    for split_name, split_recs, split_ids in [
        ("fit", fit_recordings, fit_ids),
        ("test", test_recordings_df, test_ids),
    ]:
        if len(split_recs) == 0:
            logger.warning("No vowel recordings for %s split", split_name)
            continue
        ds = VoiceFMDataset(
            recording_manifest=split_recs.reset_index(drop=True),
            participant_table=participants,
            audio_dir=audio_dir,
            task_type_map=task_type_map,
            feature_config=feature_config,
            age_mean=age_mean,
            age_std=age_std,
        )
        cats = (
            participant_strata.loc[
                [pid for pid in split_ids if pid in participant_strata.index]
            ]
            if use_cat_strat and participant_strata is not None
            else None
        )
        sampler = ParticipantBatchSampler(
            split_recs.reset_index(drop=True),
            participant_categories=cats,
            batch_size=batch_size,
            task_stratify=False,
            drop_last=False,
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_sampler=sampler,
            collate_fn=voicefm_collate_fn, num_workers=num_workers,
        )
        datasets_and_loaders[split_name] = loader

    # Build audio encoder and load checkpoint
    if "whisper" in str(checkpoint_path):
        config["model"]["audio_encoder"]["type"] = "whisper"
        config["model"]["audio_encoder"]["backbone"] = "openai/whisper-large-v2"
        config["model"]["audio_encoder"]["freeze_backbone"] = True
        config["model"]["audio_encoder"]["unfreeze_last_n"] = 4
    audio_encoder = build_audio_encoder(config["model"], len(task_type_map) + 1, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ae_state = {}
    prefix = "audio_encoder."
    for k, v in ckpt["model_state_dict"].items():
        if k.startswith(prefix):
            ae_state[k[len(prefix):]] = v
    audio_encoder.load_state_dict(ae_state)
    logger.info(
        "Loaded audio encoder from checkpoint epoch %d (val_loss=%.4f)",
        ckpt.get("epoch", -1), ckpt.get("val_loss", float("nan")),
    )

    # Extract embeddings
    embeds = {}
    for split_name in ["fit", "test"]:
        if split_name not in datasets_and_loaders:
            continue
        logger.info("Extracting VoiceFM embeddings (%s)...", split_name)
        embeds[split_name] = extract_embeddings_with_hook(
            audio_encoder, datasets_and_loaders[split_name], device,
        )

    # HuBERT baseline
    hubert_embeds = {}
    methods = ["voicefm_256d", "voicefm_768d"]
    if args.baseline:
        for split_name in ["fit", "test"]:
            if split_name not in datasets_and_loaders:
                continue
            logger.info("Extracting HuBERT baseline (%s)...", split_name)
            hubert_embeds[split_name] = extract_hubert_baseline(
                datasets_and_loaders[split_name], device,
            )
        methods.append("hubert_768d")

    # Aggregate per participant
    acoustic_agg = _agg_acoustic(acoustic_df)

    # Build aligned arrays for test set only (for PCA/heatmap)
    # For Ridge, we use all data with CV
    test_pids = list(set(embeds["test"]["participant_ids"]) & set(acoustic_agg.index))
    logger.info("Test participants with both embeddings and acoustic features: %d", len(test_pids))

    # All participants with both (for CV analysis)
    all_embed_pids = set()
    all_embeds_256d = {}
    all_embeds_768d = {}
    for split_name in ["fit", "test"]:
        if split_name not in embeds:
            continue
        agg_256 = _agg(embeds[split_name]["audio_embeds_256d"], embeds[split_name]["participant_ids"])
        agg_768 = _agg(embeds[split_name]["audio_embeds_768d"], embeds[split_name]["participant_ids"])
        all_embeds_256d.update(agg_256)
        all_embeds_768d.update(agg_768)
        all_embed_pids.update(agg_256.keys())

    # All participants with both embeddings and acoustic features
    common_pids = sorted(all_embed_pids & set(acoustic_agg.index))
    logger.info("Participants with both embeddings and acoustic: %d", len(common_pids))

    X_256d = np.array([all_embeds_256d[pid] for pid in common_pids])
    X_768d = np.array([all_embeds_768d[pid] for pid in common_pids])
    Y_acoustic = acoustic_agg.loc[common_pids][ACOUSTIC_FEATURES].values

    # HuBERT aligned arrays
    X_hubert = None
    if args.baseline:
        all_hubert = {}
        for split_name in ["fit", "test"]:
            if split_name not in hubert_embeds:
                continue
            agg = _agg(hubert_embeds[split_name]["audio_embeds"], hubert_embeds[split_name]["participant_ids"])
            all_hubert.update(agg)
        hubert_pids = sorted(set(all_hubert.keys()) & set(acoustic_agg.index))
        X_hubert = np.array([all_hubert[pid] for pid in hubert_pids])
        Y_hubert = acoustic_agg.loc[hubert_pids][ACOUSTIC_FEATURES].values

    # ── Analysis 1: PCA + Spearman correlation ────────────────────────────
    logger.info("Running PCA + Spearman correlation analysis...")
    pca_result = pca_spearman_analysis(X_256d, Y_acoustic, ACOUSTIC_FEATURES, n_components=20)

    # ── Analysis 2: Ridge regression R² (5-fold CV) ───────────────────────
    logger.info("Running Ridge regression (5-fold CV)...")
    all_metrics = {}

    m256 = ridge_cv_analysis(X_256d, Y_acoustic, ACOUSTIC_FEATURES, "voicefm_256d")
    all_metrics.update(m256)

    m768 = ridge_cv_analysis(X_768d, Y_acoustic, ACOUSTIC_FEATURES, "voicefm_768d")
    all_metrics.update(m768)

    if args.baseline and X_hubert is not None:
        m_hub = ridge_cv_analysis(X_hubert, Y_hubert, ACOUSTIC_FEATURES, "hubert_768d")
        all_metrics.update(m_hub)

    # ── Analysis 3: CCA ──────────────────────────────────────────────────
    logger.info("Running CCA...")
    cca_corrs_256 = cca_analysis(X_256d, Y_acoustic)
    cca_corrs_768 = cca_analysis(X_768d, Y_acoustic)
    all_metrics["cca/voicefm_256d"] = cca_corrs_256
    all_metrics["cca/voicefm_768d"] = cca_corrs_768
    if args.baseline and X_hubert is not None:
        cca_corrs_hub = cca_analysis(X_hubert, Y_hubert)
        all_metrics["cca/hubert_768d"] = cca_corrs_hub

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  Acoustic Feature Grounding: Ridge Regression R² (5-fold CV)")
    print(f"{'=' * 90}")

    header = f"  {'Method':<18s}  {'Feature':<18s}  {'R²':>8s}  {'Spearman':>8s}  {'MAE':>8s}  {'N':>6s}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for method in methods:
        for feat in ACOUSTIC_FEATURES:
            r2 = all_metrics.get(f"{method}/{feat}/r2")
            if r2 is None:
                continue
            rho = all_metrics.get(f"{method}/{feat}/spearman", np.nan)
            mae = all_metrics.get(f"{method}/{feat}/mae", np.nan)
            n = all_metrics.get(f"{method}/{feat}/n", 0)
            print(f"  {method:<18s}  {feat:<18s}  {r2:8.3f}  {rho:8.3f}  {mae:8.3f}  {n:6d}")
        print()

    # CCA summary
    print(f"  CCA canonical correlations:")
    for method in methods:
        corrs = all_metrics.get(f"cca/{method}", [])
        if corrs:
            corr_str = ", ".join(f"{c:.3f}" for c in corrs)
            print(f"  {method:<18s}  [{corr_str}]")
    print(f"{'=' * 90}")

    # ── Figures ──────────────────────────────────────────────────────────
    logger.info("Generating figures...")

    plot_correlation_heatmap(
        pca_result["rho_matrix"],
        pca_result["pval_matrix"],
        ACOUSTIC_FEATURES,
        pca_result["explained_variance"],
        out_dir / "correlation_heatmap.png",
    )

    plot_ridge_r2_comparison(all_metrics, methods, out_dir / "ridge_r2_comparison.png")

    # Best method for scatter
    mean_r2 = {}
    for method in methods:
        r2_vals = [all_metrics.get(f"{method}/{f}/r2", np.nan) for f in ACOUSTIC_FEATURES]
        valid = [v for v in r2_vals if not np.isnan(v)]
        mean_r2[method] = np.mean(valid) if valid else -999
    best_method = max(mean_r2, key=mean_r2.get)
    logger.info("Best method: %s (mean R²=%.3f)", best_method, mean_r2[best_method])

    plot_scatter_top_features(all_metrics, best_method, out_dir / "scatter_top_features.png")

    plot_pca_acoustic_coloring(
        pca_result["pca_components"],
        Y_acoustic,
        ACOUSTIC_FEATURES,
        out_dir / "pca_acoustic_coloring.png",
    )

    # ── Save results ─────────────────────────────────────────────────────
    json_metrics = {}
    for k, v in all_metrics.items():
        if k.endswith("/y_true") or k.endswith("/y_pred"):
            continue
        json_metrics[k] = v

    # Add PCA explained variance
    json_metrics["pca/explained_variance"] = pca_result["explained_variance"]

    # Add Spearman heatmap as nested dict
    json_metrics["spearman_heatmap"] = {
        feat: {f"PC{j+1}": float(pca_result["rho_matrix"][i, j])
               for j in range(pca_result["rho_matrix"].shape[1])}
        for i, feat in enumerate(ACOUSTIC_FEATURES)
    }

    results_path = out_dir / "acoustic_grounding_results.json"
    with open(results_path, "w") as f:
        json.dump(json_metrics, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
