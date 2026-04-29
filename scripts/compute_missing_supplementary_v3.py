#!/usr/bin/env python3
"""Compute missing supplementary figure data for VoiceFM-Whisper.

Produces: results/voicefm_whisper_supplementary.json with:
  1. Frozen Whisper 1280d acoustic grounding (Ridge R²)
  2. VoiceFM-Whisper PCA + Spearman heatmap
  3. NN retrieval: severity + voice quality differences
  4. Frozen Whisper NN retrieval for comparison
  5. Frozen HuBERT NN retrieval for comparison (from old HuBERT data)

All computed locally — no GPU needed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results_v3"

ACOUSTIC_FEATURES = [
    "f0_mean", "f0_sd", "f0_min", "f0_max", "f0_range",
    "jitter_local", "jitter_rap", "shimmer_local", "shimmer_apq3",
    "hnr", "cpps", "f1_mean", "f2_mean", "f3_mean",
]


def load_participant_embeddings(npz_path):
    """Load participant-level embeddings from npz."""
    data = np.load(npz_path, allow_pickle=True)
    pids = data["pids"]
    embs = data["embeddings"]
    return {str(p): embs[i] for i, p in enumerate(pids)}


def mean_pool_recordings(npz_path):
    """Mean-pool recording-level embeddings to participant level."""
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    participant_ids = data["participant_ids"]

    pid_to_idx = {}
    for i, pid in enumerate(participant_ids):
        pid_to_idx.setdefault(str(pid), []).append(i)

    pid_mean = {}
    for pid, idxs in pid_to_idx.items():
        pid_mean[pid] = embeddings[idxs].mean(axis=0)
    return pid_mean


def acoustic_grounding(pid_mean, acoustic_df, label="model"):
    """Ridge regression R² for predicting acoustic features from embeddings."""
    common = [p for p in pid_mean if p in acoustic_df.index]
    X = np.array([pid_mean[p] for p in common])
    results = {}

    for feat in ACOUSTIC_FEATURES:
        if feat not in acoustic_df.columns:
            continue
        y_all = acoustic_df.loc[common, feat].values.astype(float)
        valid = np.isfinite(y_all)
        if valid.sum() < 30:
            continue
        X_v, y_v = X[valid], y_all[valid]
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        y_pred = cross_val_predict(pipe, X_v, y_v, cv=5)
        ss_res = np.sum((y_v - y_pred) ** 2)
        ss_tot = np.sum((y_v - np.mean(y_v)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results[feat] = {"r2": float(r2), "n": int(valid.sum())}

    mean_r2 = np.mean([v["r2"] for v in results.values()])
    print(f"  {label}: {len(results)} features, mean R²={mean_r2:.3f}")
    return results


def pca_spearman_heatmap(pid_mean, acoustic_df, n_components=5):
    """PCA on embeddings + Spearman correlation with acoustic features."""
    common = [p for p in pid_mean if p in acoustic_df.index]
    X = np.array([pid_mean[p] for p in common])

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    explained = pca.explained_variance_ratio_.tolist()

    heatmap = {}
    for feat in ACOUSTIC_FEATURES:
        if feat not in acoustic_df.columns:
            continue
        y = acoustic_df.loc[common, feat].values.astype(float)
        valid = np.isfinite(y)
        if valid.sum() < 30:
            continue
        row = []
        for pc_i in range(n_components):
            rho, _ = scipy_stats.spearmanr(X_pca[valid, pc_i], y[valid])
            row.append(float(rho) if np.isfinite(rho) else 0.0)
        heatmap[feat] = row

    print(f"  PCA: {n_components} components, explained var = {[f'{v:.2f}' for v in explained]}")
    return {"explained_variance": explained, "heatmap": heatmap}


def nn_retrieval_full(pid_mean, participants, k=5, label="model"):
    """NN retrieval with category match, severity diff, voice quality diff."""
    pids = sorted(pid_mean.keys())
    X = np.array([pid_mean[p] for p in pids])
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm = X / norms
    sim = X_norm @ X_norm.T
    np.fill_diagonal(sim, -np.inf)

    # Disease categories
    cat_map = {}
    for pid in pids:
        if pid in participants.index:
            row = participants.loc[pid]
            for cat in ["cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]:
                if cat in row and int(row[cat]) == 1:
                    cat_map[pid] = cat
                    break
            if pid not in cat_map:
                cat_map[pid] = "control" if int(row.get("gsd_control", 0)) == 1 else "unknown"

    # Severity (VHI-10 preferred as voice-specific; falls back to PHQ-9 / GAD-7).
    # Column names in participants.parquet are *_total (not *_score); -1 is sentinel for missing.
    severity = {}
    for col in ["vhi10_total", "phq9_total", "gad7_total",
                "vhi10_score", "phq9_score", "gad7_score"]:
        if col in participants.columns:
            for pid in pids:
                if pid in participants.index:
                    val = participants.loc[pid, col]
                    if np.isfinite(val) and val >= 0:
                        severity[pid] = float(val)
            if severity:
                break

    cat_matches, total = 0, 0
    severity_diffs, quality_diffs = [], []

    for i, pid in enumerate(pids):
        if pid not in cat_map or cat_map[pid] == "unknown":
            continue
        nn_idx = np.argsort(sim[i])[-k:]
        for j in nn_idx:
            nn_pid = pids[j]
            if nn_pid not in cat_map or cat_map[nn_pid] == "unknown":
                continue
            total += 1
            if cat_map[nn_pid] == cat_map[pid]:
                cat_matches += 1
            if pid in severity and nn_pid in severity:
                severity_diffs.append(abs(severity[pid] - severity[nn_pid]))

    match_rate = cat_matches / total if total > 0 else 0
    mean_sev = float(np.mean(severity_diffs)) if severity_diffs else None

    result = {
        "cat_match_rate": float(match_rate),
        "mean_severity_diff": mean_sev,
        "k": k,
        "n_pairs": total,
    }
    print(f"  {label} NN: match={match_rate:.3f}, sev_diff={mean_sev}")
    return result


def main():
    print("Loading data...")
    acoustic_raw = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "acoustic_features.parquet")
    # acoustic_features.parquet is recording-level with record_id = participant_id
    # Mean-pool acoustic features per participant
    feat_cols = [c for c in acoustic_raw.columns if c not in ("recording_id", "record_id")]
    acoustic_df = acoustic_raw.groupby("record_id")[feat_cols].mean()
    acoustic_df.index.name = "participant_id"
    print(f"  Acoustic features: {len(acoustic_df)} participants, {len(feat_cols)} features")

    participants = pd.read_parquet(PROJECT_ROOT / "data" / "processed_v3" / "participants.parquet")
    if "participant_id" in participants.columns:
        participants = participants.set_index("participant_id")

    # Load embeddings
    print("Loading VoiceFM-Whisper embeddings...")
    vw_embs = load_participant_embeddings(RESULTS / "voicefm_whisper_embeddings.npz")
    print(f"  {len(vw_embs)} participants, {len(list(vw_embs.values())[0])}d")

    print("Loading frozen Whisper embeddings (mean-pooling recordings)...")
    fw_embs = mean_pool_recordings(RESULTS / "whisper_recording_embeddings.npz")
    print(f"  {len(fw_embs)} participants, {len(list(fw_embs.values())[0])}d")

    results = {}

    # 1. Acoustic grounding
    print("\n1. Acoustic grounding...")
    results["acoustic_grounding_vw"] = acoustic_grounding(vw_embs, acoustic_df, "VoiceFM-Whisper 256d")
    results["acoustic_grounding_fw"] = acoustic_grounding(fw_embs, acoustic_df, "Frozen Whisper 1280d")

    # 2. PCA + Spearman heatmap
    print("\n2. PCA Spearman heatmap...")
    results["pca"] = pca_spearman_heatmap(vw_embs, acoustic_df)

    # 3. NN retrieval (full)
    print("\n3. NN retrieval...")
    results["nn_vw"] = nn_retrieval_full(vw_embs, participants, label="VoiceFM-Whisper")
    results["nn_fw"] = nn_retrieval_full(fw_embs, participants, label="Frozen Whisper")

    # Save
    out_path = RESULTS / "voicefm_whisper_supplementary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
