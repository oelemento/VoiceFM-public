#!/usr/bin/env python3
"""Phase 3.7: Prospective validation on 138 held-out v3 test participants.

Trains linear probes on all 846 train participants (cohort_split=="train"),
evaluates on all 138 test participants (cohort_split=="test") that were added
to the v3.0.0 REDCap export after the v2.3.0 snapshot.

Reuses per-seed embeddings saved by unified_gsd_probes_v3.py + unified_hear_probes_v3.py
(CPU-only, seconds to run). No GPU needed.

Output: results_v3/prospective_test_probes.json
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).parent.parent
RESULTS = PROJECT / "results_v3"
SEEDS = [42, 43, 44, 45, 46]

GSD_CATS = ["gsd_control", "cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]
GSD_DIAGS = [
    "gsd_parkinsons", "gsd_alz_dementia_mci", "gsd_mtd", "gsd_copd_asthma",
    "gsd_depression", "gsd_airway_stenosis", "gsd_benign_lesion", "gsd_anxiety",
    "gsd_laryngeal_dystonia",
]
ALL_LABELS = GSD_CATS + GSD_DIAGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_probe(X_train, y_train, X_test, y_test):
    """Identical to run_probe in unified_gsd_probes_v3.py."""
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return float("nan")
    if np.sum(y_test) < 2 or np.sum(y_train) < 2:
        return float("nan")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_tr_s, y_train)
    probs = clf.predict_proba(X_te_s)[:, 1]
    return float(roc_auc_score(y_test, probs))


def load_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return {str(pid): emb for pid, emb in zip(data["pids"], data["embeddings"])}


def prospective_probes(pid_mean, participants_train, participants_test):
    train_ids = [p for p in participants_train.index if p in pid_mean]
    test_ids = [p for p in participants_test.index if p in pid_mean]
    if not train_ids or not test_ids:
        return {}
    X_train = np.array([pid_mean[p] for p in train_ids])
    X_test = np.array([pid_mean[p] for p in test_ids])
    train_df = participants_train.loc[train_ids]
    test_df = participants_test.loc[test_ids]
    results = {}
    for label in ALL_LABELS:
        if label not in participants_train.columns:
            continue
        y_tr = train_df[label].values.astype(int)
        y_te = test_df[label].values.astype(int)
        auroc = run_probe(X_train, y_tr, X_test, y_te)
        if not np.isnan(auroc):
            results[label] = auroc
    return results


def main():
    participants = pd.read_parquet(PROJECT / "data" / "processed_v3" / "participants.parquet")
    train = participants[participants["cohort_split"] == "train"].copy()
    test = participants[participants["cohort_split"] == "test"].copy()
    logger.info("Train: %d, Test: %d", len(train), len(test))

    all_results = {}
    for model in ["voicefm_whisper", "voicefm_hubert"]:
        logger.info("=== %s ===", model)
        for seed in SEEDS:
            npz = RESULTS / f"{model}_seed{seed}_embeddings.npz"
            if not npz.exists():
                logger.warning("Missing: %s", npz)
                continue
            pid_mean = load_embeddings(npz)
            probes = prospective_probes(pid_mean, train, test)
            for k, v in probes.items():
                all_results.setdefault(f"{model}/{k}", []).append(v)
            cats = [probes.get(c, float("nan")) for c in GSD_CATS]
            logger.info("  seed %d: mean=%.3f [%s]", seed, np.nanmean(cats),
                        " ".join(f"{v:.3f}" for v in cats))

    out = RESULTS / "prospective_test_probes.json"
    with out.open("w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved %s", out)

    # Summary
    logger.info("\n=== SUMMARY (prospective test: 138 held-out participants) ===")
    for model in ["voicefm_whisper", "voicefm_hubert"]:
        cat_means = [np.mean(all_results.get(f"{model}/{c}", [])) for c in GSD_CATS if all_results.get(f"{model}/{c}")]
        if cat_means:
            logger.info("%-18s  cat_mean=%.3f  n_seeds=%d", model, np.nanmean(cat_means),
                        len(all_results.get(f"{model}/gsd_control", [])))


if __name__ == "__main__":
    main()
