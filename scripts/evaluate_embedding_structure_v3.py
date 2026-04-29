#!/usr/bin/env python3
"""H18–H21: Embedding structure deep dive.

Four analyses of VoiceFM embedding space, comparing to frozen HuBERT baseline:
  H18: Nearest-neighbor retrieval — clinical similarity of nearest neighbors
  H19: Within-participant consistency — intra-person cosine similarity across tasks
  H20: Embedding UMAP — clinical structure visualization
  H21: CKA layer comparison — representation shift at each HuBERT layer

Extracts embeddings ONCE, then runs all analyses on cached arrays.

Usage:
    python3.11 scripts/evaluate_embedding_structure_v3.py \
        --checkpoint checkpoints_exp_d_hard_negatives/best_model.pt \
        --out-dir figures/embedding_structure
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio_dataset import VoiceFMDataset, build_task_type_map, voicefm_collate_fn
from src.data.clinical_encoder import ClinicalFeatureProcessor
from src.data.sampler import (
    ParticipantBatchSampler,
    build_participant_strata,
    create_participant_splits,
)
from src.models.audio_encoder import AudioEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

# Recording types for H19 (within-participant consistency)
H19_RECORDING_TYPES = ["Prolonged vowel", "Glides-Low to High", "Glides-High to Low"]


# ── Config / model loading (same pattern as evaluate_acoustic_grounding.py) ───

def deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
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
    return AudioEncoder(
        backbone=ae_cfg["backbone"],
        freeze_layers=ae_cfg["freeze_layers"],
        projection_dim=ae_cfg["projection_dim"],
        num_task_types=num_task_types + 1,
        spec_augment=False,
        gradient_checkpointing=False,
    ).to(device)


def load_checkpoint(audio_encoder, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt else ckpt["model_state_dict"]
    ae_state = {}
    for k, v in state.items():
        if k.startswith("audio_encoder."):
            ae_state[k[len("audio_encoder."):]] = v
    missing, unexpected = audio_encoder.load_state_dict(ae_state, strict=False)
    logger.info("Loaded checkpoint: %d keys, %d missing, %d unexpected",
                len(ae_state), len(missing), len(unexpected))


# ── Embedding extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_voicefm_embeddings(audio_encoder, dataloader, device):
    """Extract 256d projected + 768d pre-projection embeddings."""
    audio_encoder.eval()
    pooled_cache = []

    def hook_fn(module, input, output):
        pooled_cache.append(output.detach().cpu())

    handle = audio_encoder.pooling.register_forward_hook(hook_fn)

    results = {"embeds_256d": [], "embeds_768d": [],
               "participant_ids": [], "recording_ids": [], "task_names": []}
    try:
        n_batches = len(dataloader)
        for bi, batch in enumerate(dataloader):
            if bi % 50 == 0:
                logger.info("  VoiceFM batch %d/%d", bi, n_batches)
            audio = batch["audio_input_values"].to(device)
            mask = batch["attention_mask"].to(device)
            task_ids = batch["task_type_id"].to(device)
            embeds_256d = audio_encoder(audio, mask, task_ids)
            results["embeds_256d"].append(embeds_256d.cpu())
            results["participant_ids"].extend(batch["participant_id"])
            results["recording_ids"].extend(batch["recording_id"])
            results["task_names"].extend(batch.get("task_name", [""] * len(batch["participant_id"])))

        results["embeds_256d"] = torch.cat(results["embeds_256d"]).numpy()
        pooled_768 = torch.cat(pooled_cache)
        pooled_768 = F.normalize(pooled_768, p=2, dim=-1)
        results["embeds_768d"] = pooled_768.numpy()
    finally:
        handle.remove()

    logger.info("Extracted %d VoiceFM embeddings: 256d=%s",
                len(results["participant_ids"]), results["embeds_256d"].shape)
    return results


@torch.no_grad()
def extract_hubert_baseline(dataloader, device, backbone="facebook/hubert-base-ls960"):
    """Extract 768d mean-pooled frozen HuBERT embeddings."""
    from transformers import HubertModel
    hubert = HubertModel.from_pretrained(backbone).to(device).eval()

    results = {"embeds_768d": [], "participant_ids": [], "recording_ids": [], "task_names": []}
    n_batches = len(dataloader)
    for bi, batch in enumerate(dataloader):
        if bi % 50 == 0:
            logger.info("  HuBERT batch %d/%d", bi, n_batches)
        audio = batch["audio_input_values"].to(device)
        mask = batch["attention_mask"].to(device)
        output = hubert(input_values=audio, attention_mask=mask, return_dict=True)
        hidden = output.last_hidden_state
        frame_mask = hubert._get_feature_vector_attention_mask(hidden.shape[1], mask)
        frame_mask_f = frame_mask.unsqueeze(-1).float()
        pooled = (hidden * frame_mask_f).sum(dim=1) / frame_mask_f.sum(dim=1).clamp(min=1)
        pooled = F.normalize(pooled, p=2, dim=-1)
        results["embeds_768d"].append(pooled.cpu())
        results["participant_ids"].extend(batch["participant_id"])
        results["recording_ids"].extend(batch["recording_id"])
        results["task_names"].extend(batch.get("task_name", [""] * len(batch["participant_id"])))

    results["embeds_768d"] = torch.cat(results["embeds_768d"]).numpy()
    logger.info("Extracted %d HuBERT baseline embeddings", len(results["participant_ids"]))
    del hubert
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return results


@torch.no_grad()
def extract_per_layer_hidden_states(model_hubert, dataloader, device, max_recordings=500):
    """Extract per-layer hidden states for CKA (H21).

    Returns dict of {layer_idx: (N, D)} arrays, mean-pooled across time.
    """
    model_hubert.eval()
    layer_embeds = {}  # layer -> list of (D,) vectors
    n = 0

    for batch in dataloader:
        if n >= max_recordings:
            break
        audio = batch["audio_input_values"].to(device)
        mask = batch["attention_mask"].to(device)
        output = model_hubert(
            input_values=audio, attention_mask=mask,
            output_hidden_states=True, return_dict=True,
        )
        # output.hidden_states: tuple of 13 tensors (embedding layer + 12 transformer layers)
        frame_mask = model_hubert._get_feature_vector_attention_mask(
            output.hidden_states[0].shape[1], mask,
        )
        frame_mask_f = frame_mask.unsqueeze(-1).float()

        for li, hs in enumerate(output.hidden_states):
            pooled = (hs * frame_mask_f).sum(dim=1) / frame_mask_f.sum(dim=1).clamp(min=1)
            if li not in layer_embeds:
                layer_embeds[li] = []
            layer_embeds[li].append(pooled.cpu().numpy())

        n += audio.shape[0]

    return {li: np.concatenate(v)[:max_recordings] for li, v in layer_embeds.items()}


# ── H18: Nearest-neighbor retrieval ──────────────────────────────────────────

def analyze_nn_retrieval(vfm_embeds, hub_embeds, participant_ids, participants_df, capev_df, out_dir):
    """For each participant, find 5 NNs in VoiceFM vs HuBERT space.
    Measure clinical similarity of neighbors."""
    logger.info("H18: Nearest-neighbor retrieval analysis...")

    # Mean-pool per participant
    vfm_by_pid = {}
    hub_by_pid = {}
    for i, pid in enumerate(participant_ids):
        vfm_by_pid.setdefault(pid, []).append(vfm_embeds[i])
        hub_by_pid.setdefault(pid, []).append(hub_embeds[i])

    pids = sorted(vfm_by_pid.keys())
    vfm_mat = np.array([np.mean(vfm_by_pid[p], axis=0) for p in pids])
    hub_mat = np.array([np.mean(hub_by_pid[p], axis=0) for p in pids])

    # Normalize for cosine similarity
    vfm_mat = vfm_mat / np.linalg.norm(vfm_mat, axis=1, keepdims=True)
    hub_mat = hub_mat / np.linalg.norm(hub_mat, axis=1, keepdims=True)

    # Build clinical metadata lookup
    cat_cols = ["cat_voice", "cat_neuro", "cat_mood", "cat_respiratory", "gsd_control"]
    pid_to_cats = {}
    pid_to_severity = {}
    pid_to_vq = {}

    for pid in pids:
        row = participants_df[participants_df.index == pid]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        cats = set()
        if row.get("gsd_control", 0) == 1:
            cats.add("control")
        for c in ["cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]:
            if row.get(c, 0) == 1:
                cats.add(c)
        pid_to_cats[pid] = cats

        capev_row = capev_df[capev_df.index == pid] if pid in capev_df.index else pd.DataFrame()
        if len(capev_row) > 0:
            pid_to_severity[pid] = capev_row.iloc[0].get("overall_severity", np.nan)
            pid_to_vq[pid] = capev_row.iloc[0].get("voice_quality", np.nan)

    # Compute cosine similarity matrices
    vfm_sim = vfm_mat @ vfm_mat.T
    hub_sim = hub_mat @ hub_mat.T

    # For each participant, find top-5 NNs (excluding self)
    k = 5
    vfm_cat_match = []
    hub_cat_match = []
    vfm_sev_diff = []
    hub_sev_diff = []
    vfm_vq_diff = []
    hub_vq_diff = []

    for i, pid in enumerate(pids):
        if pid not in pid_to_cats:
            continue
        my_cats = pid_to_cats[pid]
        my_sev = pid_to_severity.get(pid, np.nan)
        my_vq = pid_to_vq.get(pid, np.nan)

        # VoiceFM NNs
        vfm_nn = np.argsort(vfm_sim[i])[::-1][1:k+1]
        hub_nn = np.argsort(hub_sim[i])[::-1][1:k+1]

        for nn_idx in vfm_nn:
            nn_pid = pids[nn_idx]
            nn_cats = pid_to_cats.get(nn_pid, set())
            vfm_cat_match.append(1 if len(my_cats & nn_cats) > 0 else 0)
            nn_sev = pid_to_severity.get(nn_pid, np.nan)
            nn_vq = pid_to_vq.get(nn_pid, np.nan)
            if not np.isnan(my_sev) and not np.isnan(nn_sev):
                vfm_sev_diff.append(abs(my_sev - nn_sev))
            if not np.isnan(my_vq) and not np.isnan(nn_vq):
                vfm_vq_diff.append(abs(my_vq - nn_vq))

        for nn_idx in hub_nn:
            nn_pid = pids[nn_idx]
            nn_cats = pid_to_cats.get(nn_pid, set())
            hub_cat_match.append(1 if len(my_cats & nn_cats) > 0 else 0)
            nn_sev = pid_to_severity.get(nn_pid, np.nan)
            nn_vq = pid_to_vq.get(nn_pid, np.nan)
            if not np.isnan(my_sev) and not np.isnan(nn_sev):
                hub_sev_diff.append(abs(my_sev - nn_sev))
            if not np.isnan(my_vq) and not np.isnan(nn_vq):
                hub_vq_diff.append(abs(my_vq - nn_vq))

    results = {
        "n_participants": len(pids),
        "k": k,
        "voicefm_cat_match_rate": float(np.mean(vfm_cat_match)),
        "hubert_cat_match_rate": float(np.mean(hub_cat_match)),
        "voicefm_mean_severity_diff": float(np.mean(vfm_sev_diff)) if vfm_sev_diff else None,
        "hubert_mean_severity_diff": float(np.mean(hub_sev_diff)) if hub_sev_diff else None,
        "voicefm_mean_vq_diff": float(np.mean(vfm_vq_diff)) if vfm_vq_diff else None,
        "hubert_mean_vq_diff": float(np.mean(hub_vq_diff)) if hub_vq_diff else None,
        "n_severity_pairs": len(vfm_sev_diff),
        "n_vq_pairs": len(vfm_vq_diff),
    }

    logger.info("H18 results: VoiceFM cat match=%.3f, HuBERT cat match=%.3f",
                results["voicefm_cat_match_rate"], results["hubert_cat_match_rate"])
    logger.info("H18 results: VoiceFM sev diff=%.1f, HuBERT sev diff=%.1f",
                results["voicefm_mean_severity_diff"] or 0, results["hubert_mean_severity_diff"] or 0)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) Category match rate
    ax = axes[0]
    bars = ax.bar(["VoiceFM", "HuBERT"],
                  [results["voicefm_cat_match_rate"], results["hubert_cat_match_rate"]],
                  color=["#2563EB", "#9CA3AF"], alpha=0.9, width=0.5)
    ax.set_ylabel("Fraction sharing diagnosis category", fontsize=10)
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, [results["voicefm_cat_match_rate"], results["hubert_cat_match_rate"]]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Category match (k=5 NN)", fontsize=11)

    # (b) Severity difference
    ax = axes[1]
    if results["voicefm_mean_severity_diff"] is not None:
        bars = ax.bar(["VoiceFM", "HuBERT"],
                      [results["voicefm_mean_severity_diff"], results["hubert_mean_severity_diff"]],
                      color=["#2563EB", "#9CA3AF"], alpha=0.9, width=0.5)
        ax.set_ylabel("Mean |severity diff| of NNs", fontsize=10)
        for bar, val in zip(bars, [results["voicefm_mean_severity_diff"], results["hubert_mean_severity_diff"]]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Overall severity distance", fontsize=11)

    # (c) Voice quality difference
    ax = axes[2]
    if results["voicefm_mean_vq_diff"] is not None:
        bars = ax.bar(["VoiceFM", "HuBERT"],
                      [results["voicefm_mean_vq_diff"], results["hubert_mean_vq_diff"]],
                      color=["#2563EB", "#9CA3AF"], alpha=0.9, width=0.5)
        ax.set_ylabel("Mean |voice quality diff| of NNs", fontsize=10)
        for bar, val in zip(bars, [results["voicefm_mean_vq_diff"], results["hubert_mean_vq_diff"]]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Voice quality distance", fontsize=11)

    plt.tight_layout()
    fig.savefig(out_dir / "h18_nn_retrieval.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_dir / "h18_nn_retrieval.png")

    return results


# ── H19: Within-participant consistency ──────────────────────────────────────

def analyze_within_participant(vfm_embeds, hub_embeds, participant_ids, recording_ids,
                               task_names, out_dir):
    """Compare intra- vs inter-person cosine similarity: VoiceFM vs HuBERT."""
    logger.info("H19: Within-participant consistency analysis...")

    # Group embeddings by participant
    vfm_by_pid = {}
    hub_by_pid = {}
    for i, pid in enumerate(participant_ids):
        vfm_by_pid.setdefault(pid, []).append(vfm_embeds[i])
        hub_by_pid.setdefault(pid, []).append(hub_embeds[i])

    # Intra-person cosine similarity (all pairwise within same participant)
    vfm_intra = []
    hub_intra = []
    for pid in vfm_by_pid:
        vfm_recs = np.array(vfm_by_pid[pid])
        hub_recs = np.array(hub_by_pid[pid])
        if len(vfm_recs) < 2:
            continue
        # Pairwise cosine sim
        vfm_norm = vfm_recs / np.linalg.norm(vfm_recs, axis=1, keepdims=True)
        hub_norm = hub_recs / np.linalg.norm(hub_recs, axis=1, keepdims=True)
        vfm_sim = vfm_norm @ vfm_norm.T
        hub_sim = hub_norm @ hub_norm.T
        # Upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(vfm_sim, dtype=bool), k=1)
        vfm_intra.extend(vfm_sim[mask].tolist())
        hub_intra.extend(hub_sim[mask].tolist())

    # Inter-person: random sample of 10,000 cross-person pairs
    rng = np.random.RandomState(42)
    pids = list(vfm_by_pid.keys())
    vfm_inter = []
    hub_inter = []
    n_inter = 10000
    for _ in range(n_inter):
        p1, p2 = rng.choice(len(pids), 2, replace=False)
        pid1, pid2 = pids[p1], pids[p2]
        i1 = rng.randint(len(vfm_by_pid[pid1]))
        i2 = rng.randint(len(vfm_by_pid[pid2]))
        v1 = vfm_by_pid[pid1][i1] / np.linalg.norm(vfm_by_pid[pid1][i1])
        v2 = vfm_by_pid[pid2][i2] / np.linalg.norm(vfm_by_pid[pid2][i2])
        vfm_inter.append(float(v1 @ v2))
        h1 = hub_by_pid[pid1][i1] / np.linalg.norm(hub_by_pid[pid1][i1])
        h2 = hub_by_pid[pid2][i2] / np.linalg.norm(hub_by_pid[pid2][i2])
        hub_inter.append(float(h1 @ h2))

    results = {
        "n_participants_with_multiple_recordings": sum(1 for p in vfm_by_pid if len(vfm_by_pid[p]) >= 2),
        "n_intra_pairs": len(vfm_intra),
        "n_inter_pairs": n_inter,
        "voicefm_intra_mean": float(np.mean(vfm_intra)),
        "voicefm_intra_std": float(np.std(vfm_intra)),
        "voicefm_inter_mean": float(np.mean(vfm_inter)),
        "voicefm_inter_std": float(np.std(vfm_inter)),
        "hubert_intra_mean": float(np.mean(hub_intra)),
        "hubert_intra_std": float(np.std(hub_intra)),
        "hubert_inter_mean": float(np.mean(hub_inter)),
        "hubert_inter_std": float(np.std(hub_inter)),
        "voicefm_separation": float(np.mean(vfm_intra)) - float(np.mean(vfm_inter)),
        "hubert_separation": float(np.mean(hub_intra)) - float(np.mean(hub_inter)),
    }

    logger.info("H19: VoiceFM intra=%.3f inter=%.3f (sep=%.3f) | HuBERT intra=%.3f inter=%.3f (sep=%.3f)",
                results["voicefm_intra_mean"], results["voicefm_inter_mean"], results["voicefm_separation"],
                results["hubert_intra_mean"], results["hubert_inter_mean"], results["hubert_separation"])

    # Plot: overlapping histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, model_name, intra, inter, color in [
        (axes[0], "VoiceFM-256d", vfm_intra, vfm_inter, "#2563EB"),
        (axes[1], "HuBERT (frozen)", hub_intra, hub_inter, "#9CA3AF"),
    ]:
        ax.hist(intra, bins=60, alpha=0.7, color=color, label="Same person", density=True)
        ax.hist(inter, bins=60, alpha=0.4, color="#EF4444", label="Different person", density=True)
        ax.axvline(np.mean(intra), color=color, linestyle="--", linewidth=1.5)
        ax.axvline(np.mean(inter), color="#EF4444", linestyle="--", linewidth=1.5)
        sep = np.mean(intra) - np.mean(inter)
        ax.set_title(f"{model_name}  (sep={sep:.3f})", fontsize=12)
        ax.set_xlabel("Cosine similarity", fontsize=11)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Density", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "h19_within_participant.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_dir / "h19_within_participant.png")

    return results


# ── H20: Embedding UMAP ─────────────────────────────────────────────────────

def analyze_umap(vfm_embeds, hub_embeds, participant_ids, participants_df, capev_df, out_dir):
    """t-SNE of participant-level embeddings, colored by clinical variables."""
    logger.info("H20: Embedding t-SNE analysis...")
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import normalize

    # Mean-pool per participant
    vfm_by_pid = {}
    hub_by_pid = {}
    for i, pid in enumerate(participant_ids):
        vfm_by_pid.setdefault(pid, []).append(vfm_embeds[i])
        hub_by_pid.setdefault(pid, []).append(hub_embeds[i])

    pids = sorted(vfm_by_pid.keys())
    vfm_mat = normalize(np.array([np.mean(vfm_by_pid[p], axis=0) for p in pids]))
    hub_mat = normalize(np.array([np.mean(hub_by_pid[p], axis=0) for p in pids]))

    # Fit t-SNE (cosine metric via pre-normalized + euclidean)
    vfm_2d = TSNE(n_components=2, perplexity=30, metric="cosine", random_state=42).fit_transform(vfm_mat)
    hub_2d = TSNE(n_components=2, perplexity=30, metric="cosine", random_state=42).fit_transform(hub_mat)

    # Build labels
    categories = []
    severities = []
    voice_qualities = []
    for pid in pids:
        row = participants_df.loc[pid] if pid in participants_df.index else pd.Series()
        if row.get("gsd_control", 0) == 1:
            categories.append("Control")
        elif row.get("cat_voice", 0) == 1:
            categories.append("Voice")
        elif row.get("cat_neuro", 0) == 1:
            categories.append("Neurological")
        elif row.get("cat_mood", 0) == 1:
            categories.append("Mood")
        elif row.get("cat_respiratory", 0) == 1:
            categories.append("Respiratory")
        else:
            categories.append("Other")

        capev_row = capev_df.loc[pid] if pid in capev_df.index else pd.Series()
        severities.append(capev_row.get("overall_severity", np.nan) if len(capev_row) > 0 else np.nan)
        voice_qualities.append(capev_row.get("voice_quality", np.nan) if len(capev_row) > 0 else np.nan)

    categories = np.array(categories)
    severities = np.array(severities, dtype=float)
    voice_qualities = np.array(voice_qualities, dtype=float)

    cat_colors = {
        "Control": "#22C55E", "Voice": "#EF4444", "Neurological": "#3B82F6",
        "Mood": "#F59E0B", "Respiratory": "#8B5CF6", "Other": "#9CA3AF",
    }

    # 2×3 grid: top row VoiceFM, bottom row HuBERT; columns: category, severity, voice_quality
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for row, (tsne_2d, model_name) in enumerate([(vfm_2d, "VoiceFM-256d"), (hub_2d, "HuBERT (frozen)")]):
        # (a) Diagnosis category
        ax = axes[row, 0]
        for cat in ["Control", "Voice", "Neurological", "Mood", "Respiratory", "Other"]:
            mask = categories == cat
            ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1], c=cat_colors[cat],
                       s=8, alpha=0.6, label=cat, rasterized=True)
        ax.set_title(f"{model_name} — Diagnosis category", fontsize=11)
        ax.legend(fontsize=7, markerscale=2, loc="best")
        ax.set_xticks([]); ax.set_yticks([])

        # (b) Overall severity
        ax = axes[row, 1]
        valid = ~np.isnan(severities)
        sc = ax.scatter(tsne_2d[valid, 0], tsne_2d[valid, 1], c=severities[valid],
                        cmap="RdYlGn_r", s=8, alpha=0.6, vmin=0, vmax=100, rasterized=True)
        # Show NaN as gray
        ax.scatter(tsne_2d[~valid, 0], tsne_2d[~valid, 1], c="#E5E7EB",
                   s=4, alpha=0.2, rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.7, label="Overall severity")
        ax.set_title(f"{model_name} — Overall severity", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

        # (c) Voice quality
        ax = axes[row, 2]
        valid_vq = ~np.isnan(voice_qualities)
        sc = ax.scatter(tsne_2d[valid_vq, 0], tsne_2d[valid_vq, 1], c=voice_qualities[valid_vq],
                        cmap="RdYlGn_r", s=8, alpha=0.6, vmin=1, vmax=10, rasterized=True)
        ax.scatter(tsne_2d[~valid_vq, 0], tsne_2d[~valid_vq, 1], c="#E5E7EB",
                   s=4, alpha=0.2, rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.7, label="Voice quality (1–10)")
        ax.set_title(f"{model_name} — Voice quality", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(out_dir / "h20_tsne.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_dir / "h20_tsne.png")

    return {"n_participants": len(pids), "n_with_severity": int(np.sum(~np.isnan(severities))),
            "n_with_vq": int(np.sum(~np.isnan(voice_qualities)))}


# ── H21: CKA layer comparison ───────────────────────────────────────────────

def linear_cka(X, Y):
    """Linear CKA (Kornblith et al., 2019)."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


def analyze_cka(audio_encoder, dataloader, device, out_dir, max_recordings=500):
    """CKA between frozen HuBERT and VoiceFM's HuBERT at each layer."""
    logger.info("H21: CKA layer comparison (max %d recordings)...", max_recordings)
    from transformers import HubertModel

    # Extract per-layer states from VoiceFM's fine-tuned HuBERT
    voicefm_hubert = audio_encoder.hubert
    voicefm_layers = extract_per_layer_hidden_states(voicefm_hubert, dataloader, device, max_recordings)
    n_used = len(voicefm_layers[0])
    logger.info("Extracted %d recordings from VoiceFM HuBERT (%d layers)", n_used, len(voicefm_layers))

    # Extract from frozen HuBERT
    frozen_hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    frozen_layers = extract_per_layer_hidden_states(frozen_hubert, dataloader, device, max_recordings)
    del frozen_hubert
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Compute CKA at each layer
    n_layers = len(voicefm_layers)
    cka_values = []
    for li in range(n_layers):
        cka = linear_cka(voicefm_layers[li], frozen_layers[li])
        cka_values.append(float(cka))
        logger.info("  Layer %d: CKA = %.4f", li, cka)

    results = {
        "n_recordings": n_used,
        "n_layers": n_layers,
        "cka_per_layer": cka_values,
        "min_cka_layer": int(np.argmin(cka_values)),
        "min_cka_value": float(np.min(cka_values)),
    }

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    layers = list(range(n_layers))
    layer_labels = ["CNN"] + [f"L{i}" for i in range(1, n_layers)]

    colors = ["#9CA3AF"] * 9 + ["#2563EB"] * 4  # CNN + 8 frozen (L1-L8) + 4 fine-tuned (L9-L12)
    if n_layers < 13:
        colors = colors[:n_layers]

    ax.bar(layers, cka_values, color=colors[:n_layers], alpha=0.9, width=0.7)
    ax.set_xticks(layers)
    ax.set_xticklabels(layer_labels[:n_layers], fontsize=9)
    ax.set_ylabel("Linear CKA (frozen vs fine-tuned)", fontsize=11)
    ax.set_xlabel("HuBERT layer", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="#D1D5DB", linestyle="--", linewidth=0.8)

    # Annotate fine-tuned layers (L9–L12 = transformer layers 8–11)
    ax.axvspan(8.5, min(n_layers - 0.5, 12.5), alpha=0.08, color="#2563EB")
    ax.text(10.5, 0.05, "fine-tuned\n(L9–L12)", ha="center", fontsize=8, color="#2563EB", alpha=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_dir / "h21_cka_layers.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_dir / "h21_cka_layers.png")

    return results


# ── Dataset building ─────────────────────────────────────────────────────────

def build_dataset_and_loader(config, recording_filter=None, batch_size=8):
    """Build dataset, optionally filtered to specific recording types."""
    data_dir = PROJECT_ROOT / "data" / "processed_v3"
    participants = pd.read_parquet(data_dir / "participants.parquet")
    recordings = pd.read_parquet(data_dir / "recordings.parquet")

    # Build task_type_map from FULL recordings (before filtering)
    task_type_map = build_task_type_map(recordings)

    if recording_filter is not None:
        recordings = recordings[recordings["recording_name"].isin(recording_filter)].copy()
        logger.info("Filtered to %d recordings (%s)", len(recordings), recording_filter)

    processor = ClinicalFeatureProcessor()
    feature_config = processor.get_feature_names()

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
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}

    # Age normalization from training participants
    train_ids = splits["train"]
    train_participants = participants[participants.index.isin(train_ids)]
    train_ages = train_participants["age"].replace(-1, float("nan")).dropna()
    age_mean = float(train_ages.mean())
    age_std = float(train_ages.std()) if train_ages.std() > 0 else 1.0

    dataset = VoiceFMDataset(
        recording_manifest=recordings.reset_index(drop=True),
        participant_table=participants,
        audio_dir=str(PROJECT_ROOT / "data" / "audio"),
        task_type_map=task_type_map,
        feature_config=feature_config,
        age_mean=age_mean,
        age_std=age_std,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=list(range(len(dataset))),
        collate_fn=voicefm_collate_fn,
        num_workers=0,
    )

    return dataset, loader, splits, task_type_map, participants


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-h19", action="store_true", help="Skip H19 (needs multi-type extraction)")
    ap.add_argument("--skip-h21", action="store_true", help="Skip H21 (CKA, needs 2x extraction)")
    ap.add_argument("--cka-max-recordings", type=int, default=500)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    config = load_config(args.experiment)
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    if "whisper" in str(checkpoint_path):
        config["model"]["audio_encoder"]["type"] = "whisper"
        config["model"]["audio_encoder"]["backbone"] = "openai/whisper-large-v2"
        config["model"]["audio_encoder"]["freeze_backbone"] = True
        config["model"]["audio_encoder"]["unfreeze_last_n"] = 4

    data_dir = PROJECT_ROOT / "data" / "processed_v3"
    participants_df = pd.read_parquet(data_dir / "participants.parquet")
    capev_df = pd.read_parquet(data_dir / "capev_scores.parquet")
    # Ensure index is record_id for lookup
    if "record_id" in participants_df.columns:
        participants_df = participants_df.set_index("record_id")
    if "record_id" in capev_df.columns:
        capev_df = capev_df.set_index("record_id")

    all_results = {}

    # ── Phase 1: Prolonged vowel embeddings (H18, H20) ──────────────────────
    logger.info("=== Phase 1: Prolonged vowel embeddings (H18, H20) ===")
    cache_path = out_dir / "vowel_embeddings_cache.npz"
    if cache_path.exists():
        logger.info("Loading cached vowel embeddings from %s", cache_path)
        cache = np.load(cache_path, allow_pickle=True)
        vfm_vowel = {
            "embeds_256d": cache["vfm_256d"],
            "embeds_768d": cache["vfm_768d"],
            "participant_ids": list(cache["participant_ids"]),
            "recording_ids": list(cache["recording_ids"]),
            "task_names": list(cache["task_names"]),
        }
        hub_vowel = {
            "embeds_768d": cache["hub_768d"],
            "participant_ids": list(cache["participant_ids"]),
        }
        # Still need task_type_map + audio_encoder for CKA
        _, vowel_loader, vowel_splits, task_type_map, _ = build_dataset_and_loader(
            config, recording_filter=["Prolonged vowel"], batch_size=2,
        )
        audio_encoder = build_audio_encoder(config["model"], len(task_type_map), device)
        load_checkpoint(audio_encoder, checkpoint_path, device)
    else:
        _, vowel_loader, vowel_splits, task_type_map, _ = build_dataset_and_loader(
            config, recording_filter=["Prolonged vowel"], batch_size=2,
        )
        audio_encoder = build_audio_encoder(config["model"], len(task_type_map), device)
        load_checkpoint(audio_encoder, checkpoint_path, device)

        vfm_vowel = extract_voicefm_embeddings(audio_encoder, vowel_loader, device)
        hub_vowel = extract_hubert_baseline(vowel_loader, device)

        np.savez(cache_path,
                 vfm_256d=vfm_vowel["embeds_256d"],
                 vfm_768d=vfm_vowel["embeds_768d"],
                 hub_768d=hub_vowel["embeds_768d"],
                 participant_ids=np.array(vfm_vowel["participant_ids"]),
                 recording_ids=np.array(vfm_vowel["recording_ids"]),
                 task_names=np.array(vfm_vowel["task_names"]))
        logger.info("Cached vowel embeddings to %s", cache_path)

    all_results["h18"] = analyze_nn_retrieval(
        vfm_vowel["embeds_256d"], hub_vowel["embeds_768d"],
        vfm_vowel["participant_ids"], participants_df, capev_df, out_dir,
    )

    all_results["h20"] = analyze_umap(
        vfm_vowel["embeds_256d"], hub_vowel["embeds_768d"],
        vfm_vowel["participant_ids"], participants_df, capev_df, out_dir,
    )

    # ── Phase 2: Multi-type embeddings (H19) ────────────────────────────────
    if not args.skip_h19:
        logger.info("=== Phase 2: Multi-type embeddings (H19) ===")
        _, multi_loader, _, _, _ = build_dataset_and_loader(
            config, recording_filter=H19_RECORDING_TYPES, batch_size=2,
        )

        vfm_multi = extract_voicefm_embeddings(audio_encoder, multi_loader, device)
        hub_multi = extract_hubert_baseline(multi_loader, device)

        all_results["h19"] = analyze_within_participant(
            vfm_multi["embeds_256d"], hub_multi["embeds_768d"],
            vfm_multi["participant_ids"], vfm_multi["recording_ids"],
            vfm_multi["task_names"], out_dir,
        )

    # ── Phase 3: CKA layer comparison (H21) ─────────────────────────────────
    if not args.skip_h21:
        logger.info("=== Phase 3: CKA layer comparison (H21) ===")
        all_results["h21"] = analyze_cka(
            audio_encoder, vowel_loader, device, out_dir,
            max_recordings=args.cka_max_recordings,
        )

    # ── Save combined results ────────────────────────────────────────────────
    results_path = out_dir / "embedding_structure_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("All results saved to %s", results_path)


if __name__ == "__main__":
    main()
