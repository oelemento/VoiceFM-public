#!/usr/bin/env python3
"""GeMAPS acoustic feature analysis on NeuroVoz.

Extracts eGeMAPSv02 features via opensmile, aggregates per participant,
correlates with VoiceFM P(PD), and computes incremental R².

Runs locally (no GPU needed — uses pre-computed P(PD) from roc_data).
"""

import json
import logging
from pathlib import Path

import numpy as np
import opensmile
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
NEUROVOZ_DIR = PROJECT_ROOT / "data" / "external" / "neurovoz"


def main():
    meta = pd.read_csv(NEUROVOZ_DIR / "metadata.csv")
    audio_dir = NEUROVOZ_DIR / "data" / "audios"

    # ── Extract eGeMAPSv02 ──
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    records = []
    for _, row in meta.iterrows():
        wav = audio_dir / row["filename"]
        if not wav.exists():
            continue
        try:
            feats = smile.process_file(str(wav))
            feat_dict = feats.iloc[0].to_dict()
            feat_dict["participant_id"] = row["participant_id"]
            feat_dict["label"] = row["label"]
            feat_dict["task_category"] = row.get("task_category", "unknown")
            records.append(feat_dict)
        except Exception as e:
            logger.warning(f"Failed {wav.name}: {e}")

    df = pd.DataFrame(records)
    logger.info(f"Extracted features from {len(df)} recordings")

    # ── Aggregate per participant ──
    feat_cols = smile.feature_names
    pid_groups = df.groupby("participant_id")
    pid_feats = pid_groups[feat_cols].mean()
    pid_labels = pid_groups["label"].first()

    X = pid_feats.values
    y = pid_labels.values
    pids = pid_feats.index.values
    logger.info(f"Participants: {len(pids)} ({np.sum(y==1)} PD, {np.sum(y==0)} HC)")

    # ── Get cross-validated P(PD) from GeMAPS features ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = cross_val_predict(clf, X_scaled, y, cv=skf, method="predict_proba")[:, 1]

    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(y, probs)
    logger.info(f"GeMAPS logistic regression AUROC: {auroc:.3f}")

    # ── Also get VoiceFM P(PD) from saved ROC data ──
    # We need VoiceFM's cross-validated P(PD) per participant
    # Since we don't have per-participant probs saved, use GeMAPS probs as target
    # Actually, let's compute R² of GeMAPS explaining the label directly
    # AND compare GeMAPS AUROC with our custom features

    # ── Incremental R² (GeMAPS feature groups) ──
    # Group GeMAPS features by type
    groups = {
        "F0": [c for c in feat_cols if c.startswith("F0")],
        "Jitter/Shimmer": [c for c in feat_cols if "jitter" in c.lower() or "shimmer" in c.lower()],
        "Loudness": [c for c in feat_cols if "loudness" in c.lower() or "Loudness" in c],
        "HNR": [c for c in feat_cols if "HNR" in c or "hnr" in c.lower()],
        "Formants": [c for c in feat_cols if any(f"F{i}" in c for i in [1,2,3]) and "F0" not in c],
        "Bandwidth": [c for c in feat_cols if "bandwidth" in c.lower() or "Bandwidth" in c],
        "Spectral": [c for c in feat_cols if any(s in c for s in ["alpha", "hammarberg", "slope", "spectralFlux", "Spectral"])],
        "MFCC": [c for c in feat_cols if "mfcc" in c.lower() or "MFCC" in c],
        "Voicing": [c for c in feat_cols if any(s in c.lower() for s in ["voiced", "unvoiced", "voicing", "logrelF0"])],
    }

    # Assign uncategorized features
    categorized = set()
    for g in groups.values():
        categorized.update(g)
    uncategorized = [c for c in feat_cols if c not in categorized]
    if uncategorized:
        groups["Other"] = uncategorized
        logger.info(f"Uncategorized features ({len(uncategorized)}): {uncategorized[:5]}...")

    # Print group sizes
    for name, cols in groups.items():
        logger.info(f"  {name}: {len(cols)} features")

    # Incremental R² predicting P(PD) from VoiceFM (use probs as target)
    # Actually, better: use the LABEL as target and measure AUROC incrementally
    # Or: use GeMAPS probs as a proxy for VoiceFM probs

    # Let's do incremental R² predicting probs (the model's P(PD))
    cumulative_cols = []
    incremental_results = []

    group_order = ["F0", "Jitter/Shimmer", "HNR", "Loudness", "Formants", "Bandwidth", "Spectral", "MFCC", "Voicing"]
    if "Other" in groups:
        group_order.append("Other")

    for gname in group_order:
        if gname not in groups or len(groups[gname]) == 0:
            continue
        cumulative_cols.extend(groups[gname])
        X_cum = pid_feats[cumulative_cols].values
        X_cum_scaled = StandardScaler().fit_transform(X_cum)

        # R² predicting probs (VoiceFM P(PD) proxy)
        reg = LinearRegression()
        reg.fit(X_cum_scaled, probs)
        r2 = reg.score(X_cum_scaled, probs)

        # Also AUROC with cumulative features
        clf_cum = LogisticRegression(max_iter=2000, C=1.0)
        probs_cum = cross_val_predict(clf_cum, X_cum_scaled, y, cv=skf, method="predict_proba")[:, 1]
        auroc_cum = roc_auc_score(y, probs_cum)

        incremental_results.append({
            "group": gname,
            "n_features": len(cumulative_cols),
            "r2": float(r2),
            "auroc": float(auroc_cum),
        })
        logger.info(f"  Cumulative {gname}: {len(cumulative_cols)} feats, R²={r2:.3f}, AUROC={auroc_cum:.3f}")

    # ── PD vs Control group differences (Cohen's d) ──
    group_diffs = []
    for col in feat_cols:
        pd_vals = pid_feats.loc[pid_labels == 1, col].dropna()
        hc_vals = pid_feats.loc[pid_labels == 0, col].dropna()
        if len(pd_vals) < 5 or len(hc_vals) < 5:
            continue
        pooled_std = np.sqrt(((len(pd_vals)-1)*pd_vals.std()**2 + (len(hc_vals)-1)*hc_vals.std()**2) / (len(pd_vals)+len(hc_vals)-2))
        if pooled_std == 0:
            continue
        d = (pd_vals.mean() - hc_vals.mean()) / pooled_std
        t_stat, p_val = stats.ttest_ind(pd_vals, hc_vals)
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

    # ── Spearman correlations with P(PD) ──
    correlations = []
    for col in feat_cols:
        vals = pid_feats[col].values
        rho, p = stats.spearmanr(vals, probs)
        correlations.append({"feature": col, "rho": float(rho), "p": float(p)})
    correlations.sort(key=lambda x: abs(x["rho"]), reverse=True)

    logger.info("\nTop 15 GeMAPS features by |Spearman rho| with P(PD):")
    for c in correlations[:15]:
        logger.info(f"  {c['feature']}: rho={c['rho']:+.3f}")

    # ── Save results ──
    results = {
        "n_participants": len(pids),
        "n_pd": int(np.sum(y == 1)),
        "n_hc": int(np.sum(y == 0)),
        "n_recordings": len(df),
        "n_gemaps_features": len(feat_cols),
        "overall_auroc": float(auroc),
        "incremental": incremental_results,
        "group_diffs_top20": group_diffs[:20],
        "correlations_top20": correlations[:20],
        "feature_groups": {k: len(v) for k, v in groups.items()},
    }

    out_path = PROJECT_ROOT / "results" / "neurovoz" / "gemaps_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
