#!/usr/bin/env python3
"""GeMAPS analysis on mPower — CPU only, no model checkpoint needed.

Uses pre-computed VoiceFM P(PD) from all_predictions.csv as target.
Measures how much of VoiceFM's P(PD) variance is explained by GeMAPS features.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import opensmile
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/mpower_gemaps"))
    parser.add_argument("--predictions", type=Path,
                        default=Path("results/longitudinal_pd/all_predictions.csv"),
                        help="Pre-computed VoiceFM P(PD) per recording")
    parser.add_argument("--max-recordings", type=int, default=0,
                        help="Max recordings (0 = all)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load VoiceFM predictions ──
    pred_path = PROJECT_ROOT / args.predictions
    preds = pd.read_csv(pred_path)
    # Filter to sustained vowels
    preds = preds[preds["recording_type"] == "sustained"].copy()
    logger.info(f"VoiceFM predictions: {len(preds)} sustained recordings")

    # ── Load metadata ──
    meta_path = PROJECT_ROOT / "data" / "mpower" / "mpower_metadata_dual.csv"
    if not meta_path.exists():
        meta_path = PROJECT_ROOT / "data" / "mpower" / "mpower_metadata.csv"
    meta = pd.read_csv(meta_path)

    # Merge predictions with metadata to get recording_id → participant_id + is_pd + prob_pd
    # preds already has participant_id, recording_id, prob_pd, is_pd
    vowel_meta = preds.copy()
    logger.info(f"mPower vowel recordings with P(PD): {len(vowel_meta)}")

    audio_dir = PROJECT_ROOT / "data" / "mpower" / "audio"

    # ── Subsample if requested ──
    if args.max_recordings > 0:
        pd_meta = vowel_meta[vowel_meta["is_pd"] == 1]
        hc_meta = vowel_meta[vowel_meta["is_pd"] == 0]
        n_per_group = args.max_recordings // 2
        pd_sample = pd_meta.sample(n=min(n_per_group, len(pd_meta)), random_state=42)
        hc_sample = hc_meta.sample(n=min(n_per_group, len(hc_meta)), random_state=42)
        sample_meta = pd.concat([pd_sample, hc_sample])
        logger.info(f"Sampled {len(sample_meta)} recordings ({len(pd_sample)} PD, {len(hc_sample)} HC)")
    else:
        sample_meta = vowel_meta
        logger.info(f"Using ALL {len(sample_meta)} recordings")

    # ── Extract GeMAPS ──
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    feat_cols = smile.feature_names

    records = []
    failed = 0
    for i, (_, row) in enumerate(sample_meta.iterrows()):
        wav = audio_dir / str(row["participant_id"]) / f"{row['recording_id']}.wav"
        if not wav.exists():
            failed += 1
            continue
        try:
            feats = smile.process_file(str(wav))
            feat_dict = feats.iloc[0].to_dict()
            feat_dict["participant_id"] = row["participant_id"]
            feat_dict["recording_id"] = row["recording_id"]
            feat_dict["is_pd"] = row["is_pd"]
            feat_dict["prob_pd"] = row["prob_pd"]
            records.append(feat_dict)
        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.warning(f"Failed {wav.name}: {e}")

        if (i + 1) % 200 == 0:
            logger.info(f"  Processed {i+1}/{len(sample_meta)} ({failed} failed)")

    df = pd.DataFrame(records)
    logger.info(f"Extracted GeMAPS from {len(df)} recordings ({failed} failed)")

    # ── Aggregate per participant ──
    pid_groups = df.groupby("participant_id")
    pid_feats = pid_groups[feat_cols].mean()
    pid_labels = pid_groups["is_pd"].first()
    pid_voicefm_prob = pid_groups["prob_pd"].mean()  # VoiceFM's P(PD) per participant

    # Drop columns that are all-NaN, then impute remaining NaNs with column median
    pid_feats = pid_feats.dropna(axis=1, how="all")
    feat_cols = [c for c in feat_cols if c in pid_feats.columns]
    pid_feats = pid_feats.fillna(pid_feats.median())

    X = pid_feats.values
    y = pid_labels.values.astype(int)
    probs = pid_voicefm_prob.values  # VoiceFM P(PD) as target
    pids = pid_feats.index.values
    logger.info(f"Participants: {len(pids)} ({np.sum(y==1)} PD, {np.sum(y==0)} HC)")
    logger.info(f"NaN check: {np.isnan(X).sum()} NaNs remaining")
    logger.info(f"VoiceFM P(PD) range: {probs.min():.3f} – {probs.max():.3f}, mean={probs.mean():.3f}")

    # ── GeMAPS classification AUROC (for comparison) ──
    # Scaler is fit inside each CV fold (sklearn Pipeline) to avoid leaking
    # test-fold statistics into the training scaler.
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gemaps_probs = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
    auroc_gemaps = roc_auc_score(y, gemaps_probs)
    auroc_voicefm = roc_auc_score(y, probs)
    logger.info(f"GeMAPS logistic regression AUROC: {auroc_gemaps:.3f}")
    logger.info(f"VoiceFM P(PD) AUROC: {auroc_voicefm:.3f}")

    # ── Incremental R² ──
    groups = {
        "F0": [c for c in feat_cols if c.startswith("F0")],
        "Jitter/Shimmer": [c for c in feat_cols if "jitter" in c.lower() or "shimmer" in c.lower()],
        "HNR": [c for c in feat_cols if "HNR" in c],
        "Loudness": [c for c in feat_cols if "loudness" in c.lower() or "Loudness" in c],
        "Formants": [c for c in feat_cols if any(f"F{i}" in c for i in [1,2,3]) and "F0" not in c],
        "Bandwidth": [c for c in feat_cols if "bandwidth" in c.lower() or "Bandwidth" in c],
        "Spectral": [c for c in feat_cols if any(s in c for s in ["alpha", "hammarberg", "slope", "spectralFlux", "Spectral"])],
        "MFCC": [c for c in feat_cols if "mfcc" in c.lower() or "MFCC" in c],
        "Voicing": [c for c in feat_cols if any(s in c.lower() for s in ["voiced", "unvoiced", "voicing", "logrelF0"])],
    }
    categorized = set()
    for g in groups.values():
        categorized.update(g)
    uncategorized = [c for c in feat_cols if c not in categorized]
    if uncategorized:
        groups["Other"] = uncategorized

    group_order = ["F0", "Jitter/Shimmer", "HNR", "Loudness", "Formants", "Bandwidth",
                   "Spectral", "MFCC", "Voicing", "Other"]

    cumulative_cols = []
    incremental_results = []

    for gname in group_order:
        if gname not in groups or len(groups[gname]) == 0:
            continue
        cumulative_cols.extend(groups[gname])
        X_cum = pid_feats[cumulative_cols].values

        # In-sample R² of OLS predicting VoiceFM's P(PD) — descriptive
        # goodness-of-fit, no CV intended.
        scaler_descr = StandardScaler().fit(X_cum)
        X_cum_descr = scaler_descr.transform(X_cum)
        reg = LinearRegression().fit(X_cum_descr, probs)
        r2 = reg.score(X_cum_descr, probs)

        # Cross-validated AUROC predicting label, scaler fit inside fold.
        pipe_cum = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ])
        probs_cum = cross_val_predict(pipe_cum, X_cum, y, cv=skf,
                                       method="predict_proba")[:, 1]
        auroc_cum = roc_auc_score(y, probs_cum)

        incremental_results.append({
            "group": gname,
            "n_features": len(cumulative_cols),
            "r2": float(r2),
            "auroc": float(auroc_cum),
        })
        logger.info(f"  Cumulative {gname}: {len(cumulative_cols)} feats, R²={r2:.3f}, AUROC={auroc_cum:.3f}")

    # ── Group differences ──
    group_diffs = []
    for col in feat_cols:
        pd_vals = pid_feats.loc[pid_labels == 1, col].dropna()
        hc_vals = pid_feats.loc[pid_labels == 0, col].dropna()
        if len(pd_vals) < 5 or len(hc_vals) < 5:
            continue
        pooled_std = np.sqrt(((len(pd_vals)-1)*pd_vals.std()**2 + (len(hc_vals)-1)*hc_vals.std()**2) /
                             (len(pd_vals)+len(hc_vals)-2))
        if pooled_std == 0:
            continue
        d = (pd_vals.mean() - hc_vals.mean()) / pooled_std
        _, p_val = stats.ttest_ind(pd_vals, hc_vals)
        group_diffs.append({
            "feature": col,
            "cohens_d": float(d),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05),
        })

    group_diffs.sort(key=lambda x: abs(x["cohens_d"]), reverse=True)

    logger.info("\nTop 15 GeMAPS features by |Cohen's d|:")
    for gd in group_diffs[:15]:
        sig = "*" if gd["significant"] else ""
        logger.info(f"  {gd['feature']}: d={gd['cohens_d']:+.3f} {sig}")

    # ── Correlations with P(PD) ──
    correlations = []
    for col in feat_cols:
        vals = pid_feats[col].values
        rho, p = stats.spearmanr(vals, probs)
        correlations.append({"feature": col, "rho": float(rho), "p": float(p)})
    correlations.sort(key=lambda x: abs(x["rho"]), reverse=True)

    logger.info("\nTop 15 GeMAPS features by |Spearman rho| with P(PD):")
    for c in correlations[:15]:
        logger.info(f"  {c['feature']}: rho={c['rho']:+.3f}")

    # ── Save ──
    results = {
        "n_participants": len(pids),
        "n_pd": int(np.sum(y == 1)),
        "n_hc": int(np.sum(y == 0)),
        "n_recordings": len(df),
        "n_gemaps_features": len(feat_cols),
        "gemaps_auroc": float(auroc_gemaps),
        "voicefm_auroc": float(auroc_voicefm),
        "incremental": incremental_results,
        "group_diffs_top20": group_diffs[:20],
        "correlations_top20": correlations[:20],
    }

    out_path = args.out_dir / "gemaps_mpower_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
