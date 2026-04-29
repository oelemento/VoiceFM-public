#!/usr/bin/env python3
"""Extract acoustic features from NeuroVoz and analyze what drives PD predictions.

Replicates the mPower acoustic analysis (scripts/acoustic_pd_analysis.py) for NeuroVoz:
  1. Extract Praat acoustic features from all NeuroVoz recordings
  2. Extract VoiceFM-256d and HuBERT-768d embeddings
  3. Fit 5-fold CV logistic probe to get cross-validated P(PD) per participant
  4. Correlate acoustic features with P(PD) — what does the model capture?
  5. Incremental R² — how much of the model's signal is explained by classic features?

All analysis is participant-level (mean-pool features and embeddings per participant).

Usage:
    python scripts/acoustic_neurovoz_analysis_v3.py \
        --checkpoint checkpoints_exp_d_gsd_v3_seed42/best_model.pt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
import torch
import torch.nn.functional as F
import yaml
from parselmouth.praat import call
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import HubertModel

from src.data.external_datasets import NeuroVozDataset, external_collate_fn
from src.models import build_audio_encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
NEUROVOZ_DIR = PROJECT_ROOT / "data" / "external" / "neurovoz"


def extract_acoustic_features(audio_path):
    """Extract acoustic features from a single audio file using Praat."""
    try:
        snd = parselmouth.Sound(str(audio_path))
    except Exception:
        return None

    if snd.duration < 0.5:
        return None

    try:
        snd = snd.extract_part(
            from_time=max(0, snd.get_start_time()),
            to_time=min(snd.get_end_time(), snd.get_start_time() + 30),
        )
    except Exception:
        pass

    features = {"duration": snd.duration}

    # Pitch (F0)
    try:
        pitch = call(snd, "To Pitch", 0.0, 75, 500)
        f0_values = [call(pitch, "Get value at time", t, "Hertz", "Linear")
                     for t in np.linspace(snd.get_start_time(), snd.get_end_time(), 100)]
        f0_values = [v for v in f0_values if v == v and v > 0]
        if len(f0_values) > 5:
            features["f0_mean"] = np.mean(f0_values)
            features["f0_std"] = np.std(f0_values)
            features["f0_range"] = np.max(f0_values) - np.min(f0_values)
            features["f0_cv"] = np.std(f0_values) / np.mean(f0_values)
        else:
            return None
    except Exception:
        return None

    # Jitter & Shimmer
    try:
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        features["jitter_local"] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features["jitter_rap"] = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        features["shimmer_local"] = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        features["shimmer_apq3"] = call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        features["jitter_local"] = np.nan
        features["jitter_rap"] = np.nan
        features["shimmer_local"] = np.nan
        features["shimmer_apq3"] = np.nan

    # HNR
    try:
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        features["hnr"] = call(harmonicity, "Get mean", 0, 0)
    except Exception:
        features["hnr"] = np.nan

    # Intensity
    try:
        intensity = call(snd, "To Intensity", 75, 0)
        features["intensity_mean"] = call(intensity, "Get mean", 0, 0, "dB")
        features["intensity_std"] = call(intensity, "Get standard deviation", 0, 0)
    except Exception:
        features["intensity_mean"] = np.nan
        features["intensity_std"] = np.nan

    # Formants (F1-F3)
    try:
        formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        for fi in range(1, 4):
            f_vals = []
            b_vals = []
            for t in np.linspace(snd.get_start_time(), snd.get_end_time(), 50):
                fv = call(formant, "Get value at time", fi, t, "Hertz", "Linear")
                bv = call(formant, "Get bandwidth at time", fi, t, "Hertz", "Linear")
                if fv == fv and fv > 0:
                    f_vals.append(fv)
                if bv == bv and bv > 0:
                    b_vals.append(bv)
            features[f"f{fi}_mean"] = np.mean(f_vals) if f_vals else np.nan
            features[f"f{fi}_bw"] = np.mean(b_vals) if b_vals else np.nan
        if not np.isnan(features.get("f1_mean", np.nan)) and not np.isnan(features.get("f3_mean", np.nan)):
            features["formant_dispersion"] = (features["f3_mean"] - features["f1_mean"]) / 2
        else:
            features["formant_dispersion"] = np.nan
    except Exception:
        for fi in range(1, 4):
            features[f"f{fi}_mean"] = np.nan
            features[f"f{fi}_bw"] = np.nan
        features["formant_dispersion"] = np.nan

    # MFCCs
    try:
        mfcc_obj = call(snd, "To MFCC", 13, 0.015, 0.005, 100.0, 100.0, 0.0)
        mfcc_mat = call(mfcc_obj, "To Matrix")
        mfcc_matrix = mfcc_mat.values
        n_coeffs, n_frames = mfcc_matrix.shape
        for c in range(min(n_coeffs, 13)):
            features[f"mfcc{c}_mean"] = float(np.mean(mfcc_matrix[c]))
            features[f"mfcc{c}_std"] = float(np.std(mfcc_matrix[c]))
        if n_frames > 2:
            delta = np.diff(mfcc_matrix, axis=1)
            for c in range(min(n_coeffs, 13)):
                features[f"mfcc{c}_delta_mean"] = float(np.mean(delta[c]))
                features[f"mfcc{c}_delta_std"] = float(np.std(delta[c]))
            if n_frames > 3:
                delta2 = np.diff(delta, axis=1)
                for c in range(min(n_coeffs, 13)):
                    features[f"mfcc{c}_dd_mean"] = float(np.mean(delta2[c]))
                    features[f"mfcc{c}_dd_std"] = float(np.std(delta2[c]))
    except Exception:
        for c in range(13):
            for suffix in ["mean", "std", "delta_mean", "delta_std", "dd_mean", "dd_std"]:
                features[f"mfcc{c}_{suffix}"] = np.nan

    # Spectral shape
    try:
        samples = snd.values[0]
        sr = int(snd.sampling_frequency)
        n_fft = min(len(samples), 2 * sr)
        fft_vals = np.fft.rfft(samples[:n_fft])
        powers = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        freqs = freqs[1:]
        powers = powers[1:]
        total_power = np.sum(powers)
        if total_power > 0 and len(freqs) > 10:
            prob = powers / total_power
            features["spectral_centroid"] = np.sum(freqs * prob)
            features["spectral_spread"] = np.sqrt(np.sum(((freqs - features["spectral_centroid"]) ** 2) * prob))
            spread = features["spectral_spread"]
            if spread > 0:
                diff = freqs - features["spectral_centroid"]
                features["spectral_skewness"] = np.sum((diff ** 3) * prob) / (spread ** 3)
                features["spectral_kurtosis"] = np.sum((diff ** 4) * prob) / (spread ** 4)
            else:
                features["spectral_skewness"] = np.nan
                features["spectral_kurtosis"] = np.nan
            features["spectral_flatness"] = np.exp(np.mean(np.log(powers + 1e-12))) / (np.mean(powers) + 1e-12)
            db_values = 10 * np.log10(powers + 1e-12)
            slope, _, _, _, _ = stats.linregress(freqs, db_values)
            features["spectral_slope"] = slope
        else:
            raise ValueError("No spectral power")
    except Exception:
        for k in ["spectral_centroid", "spectral_spread", "spectral_skewness",
                   "spectral_kurtosis", "spectral_flatness", "spectral_slope"]:
            features[k] = np.nan

    return features


def extract_voicefm_embeddings(audio_encoder, dataloader, device):
    """Extract 256d embeddings + participant_ids."""
    audio_encoder.eval()
    all_embs, all_labels, all_pids = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            audio = batch["audio_values"].to(device)
            mask = batch["attention_mask"].to(device)
            task_ids = torch.zeros(audio.shape[0], dtype=torch.long, device=device)
            embeds = audio_encoder(
                audio_input_values=audio, attention_mask=mask, task_type_ids=task_ids,
            )
            all_embs.append(embeds.cpu().numpy())
            all_labels.extend(batch["labels"]["disease"].numpy())
            all_pids.extend(batch["participant_ids"])
    return np.concatenate(all_embs), np.array(all_labels), all_pids


def extract_hubert_embeddings(hubert, dataloader, device):
    """Extract masked mean-pooled 768d embeddings."""
    hubert.eval()
    all_embs, all_labels, all_pids = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            audio = batch["audio_values"].to(device)
            mask = batch["attention_mask"].to(device)
            out = hubert(audio, attention_mask=mask, return_dict=True)
            hidden = out.last_hidden_state
            frame_mask = hubert._get_feature_vector_attention_mask(hidden.shape[1], mask)
            frame_mask_f = frame_mask.unsqueeze(-1).float()
            pooled = (hidden * frame_mask_f).sum(dim=1) / frame_mask_f.sum(dim=1).clamp(min=1)
            pooled = F.normalize(pooled, p=2, dim=-1)
            all_embs.append(pooled.cpu().numpy())
            all_labels.extend(batch["labels"]["disease"].numpy())
            all_pids.extend(batch["participant_ids"])
    return np.concatenate(all_embs), np.array(all_labels), all_pids


def aggregate_to_participants(values, labels, pids):
    """Mean-pool per participant."""
    pid_to_vals = {}
    pid_to_label = {}
    for val, label, pid in zip(values, labels, pids):
        if pid not in pid_to_vals:
            pid_to_vals[pid] = []
            pid_to_label[pid] = label
        pid_to_vals[pid].append(val)
    ordered_pids = sorted(pid_to_vals.keys())
    X = np.array([np.mean(pid_to_vals[p], axis=0) for p in ordered_pids])
    y = np.array([pid_to_label[p] for p in ordered_pids])
    return X, y, ordered_pids


def get_cv_probabilities(X, y):
    """Get cross-validated P(PD) for each participant using 5-fold CV."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = cross_val_predict(clf, X_scaled, y, cv=skf, method="predict_proba")
    return probs[:, 1]  # P(PD)


def incremental_r2(X_features, y_target, feature_groups):
    """Compute incremental R² as feature groups are added."""
    from sklearn.linear_model import LinearRegression

    results = []
    cumulative_cols = []
    for group_name, cols in feature_groups:
        valid_cols = [c for c in cols if c in X_features.columns]
        if not valid_cols:
            continue
        cumulative_cols.extend(valid_cols)
        X = X_features[cumulative_cols].dropna(axis=0)
        idx = X.index
        y = y_target.loc[idx]
        if len(X) < 20:
            continue
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        reg = LinearRegression()
        reg.fit(X_scaled, y)
        r2 = reg.score(X_scaled, y)
        results.append({
            "group": group_name,
            "n_features": len(cumulative_cols),
            "r2": r2,
        })
    # Compute delta R²
    for i, r in enumerate(results):
        r["delta_r2"] = r["r2"] if i == 0 else r["r2"] - results[i - 1]["r2"]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("results_v3/neurovoz"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta_csv = NEUROVOZ_DIR / "metadata.csv"
    audio_dir = NEUROVOZ_DIR / "data" / "audios"

    # Load metadata for acoustic extraction
    meta_df = pd.read_csv(meta_csv)
    logger.info("NeuroVoz metadata: %d recordings, %d participants",
                len(meta_df), meta_df["participant_id"].nunique())

    # ── Step 1: Extract acoustic features ──
    logger.info("Extracting acoustic features...")
    acoustic_rows = []
    for i, row in meta_df.iterrows():
        audio_path = audio_dir / row["filename"]
        if not audio_path.exists():
            continue
        feats = extract_acoustic_features(str(audio_path))
        if feats is None:
            continue
        feats["participant_id"] = str(row["participant_id"])
        feats["label"] = int(row["label"])
        feats["task_category"] = row.get("task_category", "unknown")
        feats["filename"] = row["filename"]
        acoustic_rows.append(feats)
        if len(acoustic_rows) % 200 == 0:
            logger.info("  %d / %d recordings processed", len(acoustic_rows), len(meta_df))

    acoustic_df = pd.DataFrame(acoustic_rows)
    acoustic_df.to_csv(args.out_dir / "acoustic_features_neurovoz.csv", index=False)
    logger.info("Extracted acoustic features for %d recordings", len(acoustic_df))

    # Aggregate acoustic features per participant
    acoustic_cols = [c for c in acoustic_df.columns
                     if c not in ("participant_id", "label", "task_category", "filename")]
    acoustic_ptcp = acoustic_df.groupby("participant_id").agg(
        {**{c: "mean" for c in acoustic_cols}, "label": "first"}
    ).reset_index()
    logger.info("Participant-level acoustic: %d participants", len(acoustic_ptcp))

    # ── Step 2: Load models and extract embeddings ──
    config = {}
    for name in ["model", "data", "train"]:
        cfg_path = PROJECT_ROOT / "configs" / f"{name}.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                config[name] = yaml.safe_load(f)

    logger.info("Loading VoiceFM checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    task_emb_key = "audio_encoder.task_embedding.weight"
    num_task_types = state[task_emb_key].shape[0] if task_emb_key in state else 100
    audio_encoder = build_audio_encoder(
        config=config["model"]["audio_encoder"], num_task_types=num_task_types,
    )
    ae_state = {k.replace("audio_encoder.", "", 1): v for k, v in state.items()
                if k.startswith("audio_encoder.")}
    audio_encoder.load_state_dict(ae_state)
    audio_encoder = audio_encoder.to(device).eval()

    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()

    # Use "all" category dataset
    ds = NeuroVozDataset(meta_csv, audio_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                       collate_fn=external_collate_fn, num_workers=2)

    vfm_emb, vfm_labels, vfm_pids = extract_voicefm_embeddings(audio_encoder, loader, device)
    hub_emb, hub_labels, hub_pids = extract_hubert_embeddings(hubert, loader, device)

    # Aggregate embeddings per participant
    X_vfm, y_vfm, pids_vfm = aggregate_to_participants(vfm_emb, vfm_labels, vfm_pids)
    X_hub, y_hub, pids_hub = aggregate_to_participants(hub_emb, hub_labels, hub_pids)

    logger.info("Participant embeddings: VFM %s, HuBERT %s", X_vfm.shape, X_hub.shape)

    # ── Step 3: Get cross-validated P(PD) ──
    prob_vfm = get_cv_probabilities(X_vfm, y_vfm)
    prob_hub = get_cv_probabilities(X_hub, y_hub)

    # Create participant-level dataframe with P(PD)
    ptcp_df = pd.DataFrame({
        "participant_id": pids_vfm,
        "label": y_vfm,
        "prob_pd_vfm": prob_vfm,
        "prob_pd_hub": prob_hub,
    })

    # Merge with acoustic features
    merged = ptcp_df.merge(acoustic_ptcp.drop(columns=["label"]),
                           on="participant_id", how="inner")
    logger.info("Merged: %d participants with both embeddings and acoustic features", len(merged))

    # ── Step 4: Correlations ──
    classic_cols = ["f0_mean", "f0_std", "f0_range", "f0_cv", "jitter_local", "jitter_rap",
                    "shimmer_local", "shimmer_apq3", "hnr", "intensity_mean", "intensity_std",
                    "duration"]
    formant_cols = ["f1_mean", "f2_mean", "f3_mean", "f1_bw", "f2_bw", "f3_bw",
                    "formant_dispersion"]
    spectral_cols = ["spectral_centroid", "spectral_spread", "spectral_skewness",
                     "spectral_kurtosis", "spectral_flatness", "spectral_slope"]
    mfcc_mean_cols = [f"mfcc{c}_mean" for c in range(13)]
    mfcc_std_cols = [f"mfcc{c}_std" for c in range(13)]
    mfcc_delta_cols = [f"mfcc{c}_delta_mean" for c in range(13)]
    mfcc_delta_std_cols = [f"mfcc{c}_delta_std" for c in range(13)]
    mfcc_dd_cols = [f"mfcc{c}_dd_mean" for c in range(13)]
    mfcc_dd_std_cols = [f"mfcc{c}_dd_std" for c in range(13)]

    print("\n" + "=" * 70)
    print("ACOUSTIC FEATURE CORRELATIONS WITH P(PD)")
    print("=" * 70)

    correlations = {}
    for target_name, target_col in [("VoiceFM P(PD)", "prob_pd_vfm"),
                                      ("HuBERT P(PD)", "prob_pd_hub")]:
        print(f"\n--- {target_name} ---")
        print(f"{'Feature':<25} {'Spearman rho':>12} {'p-value':>12}")
        print("-" * 52)
        corr_list = []
        for col in classic_cols + formant_cols:
            valid = merged[[col, target_col]].dropna()
            if len(valid) < 10:
                continue
            rho, p = stats.spearmanr(valid[col], valid[target_col])
            corr_list.append({"feature": col, "rho": rho, "p": p})

        corr_list.sort(key=lambda x: abs(x["rho"]), reverse=True)
        for c in corr_list:
            sig = "***" if c["p"] < 0.001 else "**" if c["p"] < 0.01 else "*" if c["p"] < 0.05 else ""
            print(f"{c['feature']:<25} {c['rho']:>+12.3f} {c['p']:>12.4f} {sig}")
        correlations[target_name] = corr_list

    # ── Step 5: Incremental R² ──
    print("\n" + "=" * 70)
    print("INCREMENTAL R² PREDICTING P(PD) FROM ACOUSTIC FEATURES")
    print("=" * 70)

    all_mfcc = (mfcc_mean_cols + mfcc_std_cols + mfcc_delta_cols
                + mfcc_delta_std_cols + mfcc_dd_cols + mfcc_dd_std_cols)
    feature_groups = [
        ("Classic", classic_cols),
        ("+ Formants", formant_cols),
        ("+ Spectral", spectral_cols),
        ("+ MFCC means", mfcc_mean_cols),
        ("+ MFCC stds", mfcc_std_cols),
        ("+ MFCC deltas", mfcc_delta_cols + mfcc_delta_std_cols + mfcc_dd_cols + mfcc_dd_std_cols),
    ]

    inc_results = {}
    for target_name, target_col in [("VoiceFM P(PD)", "prob_pd_vfm"),
                                      ("HuBERT P(PD)", "prob_pd_hub")]:
        print(f"\n--- {target_name} ---")
        print(f"{'Feature set':<25} {'# features':>10} {'R²':>8} {'Delta R²':>10}")
        print("-" * 56)
        results = incremental_r2(
            merged[classic_cols + formant_cols + spectral_cols + all_mfcc],
            merged[target_col],
            feature_groups,
        )
        for r in results:
            print(f"{r['group']:<25} {r['n_features']:>10} {r['r2']:>8.3f} {r['delta_r2']:>+10.3f}")
        inc_results[target_name] = results

    # ── Step 6: PD vs Control group differences ──
    print("\n" + "=" * 70)
    print("PD vs CONTROL GROUP DIFFERENCES (participant-level)")
    print("=" * 70)
    print(f"{'Feature':<25} {'PD mean':>10} {'Ctrl mean':>10} {'Effect (d)':>10} {'p-value':>10}")
    print("-" * 68)

    pd_mask = merged["label"] == 1
    group_diffs = []
    for col in classic_cols + formant_cols:
        pd_vals = merged.loc[pd_mask, col].dropna()
        ctrl_vals = merged.loc[~pd_mask, col].dropna()
        if len(pd_vals) < 5 or len(ctrl_vals) < 5:
            continue
        t_stat, p_val = stats.ttest_ind(pd_vals, ctrl_vals)
        pooled_std = np.sqrt((pd_vals.std()**2 + ctrl_vals.std()**2) / 2)
        d = (pd_vals.mean() - ctrl_vals.mean()) / pooled_std if pooled_std > 0 else 0
        group_diffs.append({
            "feature": col, "pd_mean": pd_vals.mean(), "ctrl_mean": ctrl_vals.mean(),
            "cohen_d": d, "p": p_val,
        })

    group_diffs.sort(key=lambda x: abs(x["cohen_d"]), reverse=True)
    for g in group_diffs:
        sig = "***" if g["p"] < 0.001 else "**" if g["p"] < 0.01 else "*" if g["p"] < 0.05 else ""
        print(f"{g['feature']:<25} {g['pd_mean']:>10.3f} {g['ctrl_mean']:>10.3f} "
              f"{g['cohen_d']:>+10.3f} {g['p']:>10.4f} {sig}")

    # ── Save results ──
    summary = {
        "n_participants": len(merged),
        "n_pd": int(pd_mask.sum()),
        "n_hc": int((~pd_mask).sum()),
        "correlations": correlations,
        "incremental_r2": inc_results,
        "group_differences": group_diffs,
        "mean_prob_pd_vfm": {"pd": float(merged.loc[pd_mask, "prob_pd_vfm"].mean()),
                              "hc": float(merged.loc[~pd_mask, "prob_pd_vfm"].mean())},
        "mean_prob_hub": {"pd": float(merged.loc[pd_mask, "prob_pd_hub"].mean()),
                           "hc": float(merged.loc[~pd_mask, "prob_pd_hub"].mean())},
    }

    out_path = args.out_dir / "acoustic_analysis_neurovoz.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved results to %s", out_path)

    merged.to_csv(args.out_dir / "participant_features_neurovoz.csv", index=False)
    logger.info("Saved participant-level data to %s",
                args.out_dir / "participant_features_neurovoz.csv")


if __name__ == "__main__":
    main()
