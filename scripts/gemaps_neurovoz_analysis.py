#!/usr/bin/env python3
"""GeMAPS acoustic feature analysis on NeuroVoz — leakage-free version.

Differences from gemaps_neurovoz_analysis.py:
  * StandardScaler is fit **inside** each CV fold (sklearn Pipeline), not on
    the full dataset before splitting. The earlier script leaked test-fold
    statistics into training, which inflated AUROC by a small but
    detectable margin.
  * Saves a separate JSON (gemaps_analysis_clean.json) so leaky and clean
    numbers can be compared side-by-side.
"""

import json
import logging
from pathlib import Path

import numpy as np
import opensmile
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
NEUROVOZ_DIR = PROJECT_ROOT / "data" / "external" / "neurovoz"
OUT_DIR = PROJECT_ROOT / "results_v3" / "neurovoz"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def main():
    meta = pd.read_csv(NEUROVOZ_DIR / "metadata.csv")
    audio_dir = NEUROVOZ_DIR / "data" / "audios"

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
            records.append(feat_dict)
        except Exception as e:
            logger.warning(f"Failed {wav.name}: {e}")

    df = pd.DataFrame(records)
    feat_cols = smile.feature_names

    pid_groups = df.groupby("participant_id")
    pid_feats = pid_groups[feat_cols].mean()
    pid_labels = pid_groups["label"].first()

    X = pid_feats.values
    y = pid_labels.values
    pids = pid_feats.index.values
    logger.info(f"Participants: {len(pids)} ({np.sum(y==1)} PD, {np.sum(y==0)} HC)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = cross_val_predict(make_pipeline(), X, y, cv=skf, method="predict_proba")[:, 1]
    auroc_overall = roc_auc_score(y, probs)
    logger.info(f"GeMAPS LR AUROC (clean, fold-internal scaling): {auroc_overall:.3f}")

    groups = {
        "F0": [c for c in feat_cols if c.startswith("F0")],
        "Jitter/Shimmer": [c for c in feat_cols if "jitter" in c.lower() or "shimmer" in c.lower()],
        "Loudness": [c for c in feat_cols if "loudness" in c.lower() or "Loudness" in c],
        "HNR": [c for c in feat_cols if "HNR" in c or "hnr" in c.lower()],
        "Formants": [c for c in feat_cols if any(f"F{i}" in c for i in [1, 2, 3]) and "F0" not in c],
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

    group_order = ["F0", "Jitter/Shimmer", "HNR", "Loudness", "Formants",
                   "Bandwidth", "Spectral", "MFCC", "Voicing"]
    if "Other" in groups:
        group_order.append("Other")

    cumulative_cols = []
    incremental_results = []
    for gname in group_order:
        if gname not in groups or not groups[gname]:
            continue
        cumulative_cols.extend(groups[gname])
        X_cum = pid_feats[cumulative_cols].values

        # In-sample R² of OLS predicting `probs` (a goodness-of-fit
        # description, not generalization, so no CV needed).
        scaler_descr = StandardScaler().fit(X_cum)
        X_descr = scaler_descr.transform(X_cum)
        reg = LinearRegression().fit(X_descr, probs)
        r2 = reg.score(X_descr, probs)

        # Cross-validated AUROC predicting label (clean, scaler in pipeline).
        probs_cum = cross_val_predict(make_pipeline(), X_cum, y, cv=skf,
                                      method="predict_proba")[:, 1]
        auroc_cum = roc_auc_score(y, probs_cum)

        incremental_results.append({
            "group": gname,
            "n_features": len(cumulative_cols),
            "r2": float(r2),
            "auroc": float(auroc_cum),
        })
        logger.info(f"  Cumulative {gname}: {len(cumulative_cols)} feats, R²={r2:.3f}, AUROC={auroc_cum:.3f}")

    group_diffs = []
    for col in feat_cols:
        pd_vals = pid_feats.loc[pid_labels == 1, col].dropna()
        hc_vals = pid_feats.loc[pid_labels == 0, col].dropna()
        if len(pd_vals) < 5 or len(hc_vals) < 5:
            continue
        pooled_std = np.sqrt(((len(pd_vals) - 1) * pd_vals.std() ** 2 +
                              (len(hc_vals) - 1) * hc_vals.std() ** 2) /
                             (len(pd_vals) + len(hc_vals) - 2))
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

    correlations = []
    for col in feat_cols:
        rho, p = stats.spearmanr(pid_feats[col].values, probs)
        correlations.append({"feature": col, "rho": float(rho), "p": float(p)})
    correlations.sort(key=lambda x: abs(x["rho"]), reverse=True)

    results = {
        "n_participants": len(pids),
        "n_pd": int(np.sum(y == 1)),
        "n_hc": int(np.sum(y == 0)),
        "n_recordings": len(df),
        "n_gemaps_features": len(feat_cols),
        "overall_auroc": float(auroc_overall),
        "cv_strategy": "5-fold StratifiedKFold, scaler fit inside fold (sklearn Pipeline)",
        "incremental": incremental_results,
        "group_diffs_top20": group_diffs[:20],
        "correlations_top20": correlations[:20],
        "feature_groups": {k: len(v) for k, v in groups.items()},
    }

    out_path = OUT_DIR / "gemaps_analysis_clean.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {out_path}")

    # Also print a comparison vs the leaky run for the user.
    leaky_path = OUT_DIR / "gemaps_analysis.json"
    if leaky_path.exists():
        with open(leaky_path) as f:
            leaky = json.load(f)
        print("\n=== Clean vs leaky comparison ===")
        print(f"  overall_auroc: leaky={leaky['overall_auroc']:.3f}, clean={auroc_overall:.3f}, "
              f"Δ={auroc_overall - leaky['overall_auroc']:+.3f}")
        leaky_inc = {r["group"]: r for r in leaky["incremental"]}
        clean_inc = {r["group"]: r for r in incremental_results}
        for g in group_order:
            if g in leaky_inc and g in clean_inc:
                print(f"  {g:18s} AUROC: leaky={leaky_inc[g]['auroc']:.3f}, "
                      f"clean={clean_inc[g]['auroc']:.3f}, "
                      f"Δ={clean_inc[g]['auroc'] - leaky_inc[g]['auroc']:+.3f}  |  "
                      f"R²: leaky={leaky_inc[g]['r2']:.3f}, "
                      f"clean={clean_inc[g]['r2']:.3f}")


if __name__ == "__main__":
    main()
