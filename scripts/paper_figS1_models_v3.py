#!/usr/bin/env python3
"""Figure S1: Full model comparison on GSD.

Grouped bar chart: 5 models × 5 categories.
VoiceFM-Whisper, VoiceFM-HuBERT, VoiceFM-HeAR, Frozen Whisper, Frozen HuBERT.

Usage:
    python scripts/paper_figS1_models_v3.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results_v3"
PAPER_DIR = PROJECT_ROOT / "paper_v3"
PAPER_DIR.mkdir(exist_ok=True)

TASKS = ["gsd_control", "cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]
LABELS = ["Control vs\nDisease", "Voice\nDisorder", "Neuro-\nlogical", "Mood\nDisorder", "Respiratory\nDisorder"]


def compute_demo_only_aurocs():
    """Per-task AUROC using ONLY [age, is_male] features. 5 seeds (random_state 42-46)."""
    parts = pd.read_parquet(PROJECT_ROOT / "data" / "processed_v3" / "participants.parquet")
    age = parts["age"].fillna(parts["age"].median()).values.astype(float)
    gender = parts["gender"].fillna(0).values.astype(int)
    demo = np.column_stack([age, (gender == 1).astype(float)])

    out = {}
    for t in TASKS:
        if t not in parts.columns:
            out[t] = []
            continue
        y = parts[t].values.astype(int)
        if y.sum() < 10:
            out[t] = []
            continue
        seed_aurocs = []
        for seed in range(42, 47):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            fold_aurocs = []
            for tr, te in skf.split(demo, y):
                sc = StandardScaler()
                Xtr = sc.fit_transform(demo[tr]); Xte = sc.transform(demo[te])
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[tr])
                prob = clf.predict_proba(Xte)[:, 1]
                fold_aurocs.append(roc_auc_score(y[te], prob))
            seed_aurocs.append(float(np.mean(fold_aurocs)))
        out[t] = seed_aurocs
    return out


def main():
    # All models from unified_gsd_probes.json (populated by unified_gsd_probes.py + unified_hear_probes.py)
    with open(RESULTS / "unified_gsd_probes.json") as f:
        unified = json.load(f)

    # Compute demo-only baseline (age + gender)
    print("Computing demo-only baseline (5 seeds × 5-fold CV)...")
    demo_aurocs = compute_demo_only_aurocs()

    # HeAR models: load from unified_gsd_probes.json (will be populated by unified_hear_probes.py)
    models = {
        "VoiceFM-Whisper": {t: unified[f"voicefm_whisper/{t}"] for t in TASKS},
        "VoiceFM-HuBERT": {t: unified[f"voicefm_hubert/{t}"] for t in TASKS},
        "VoiceFM-HeAR": {t: unified[f"hear_voicefm/{t}"] for t in TASKS if f"hear_voicefm/{t}" in unified},
        "Frozen Whisper": {t: unified[f"frozen_whisper/{t}"] for t in TASKS},
        "Frozen HuBERT": {t: unified[f"frozen_hubert/{t}"] for t in TASKS},
        "Frozen HeAR": {t: unified[f"frozen_hear/{t}"] for t in TASKS if f"frozen_hear/{t}" in unified},
        "Demo only (age + gender)": demo_aurocs,
    }
    # Remove models with no data yet (HeAR may not be in JSON until cluster job completes)
    models = {k: v for k, v in models.items() if v}
    # Color palette
    all_colors = {"VoiceFM-Whisper": "#2563EB", "VoiceFM-HuBERT": "#7C3AED",
                  "VoiceFM-HeAR": "#D97706", "Frozen Whisper": "#6B7280",
                  "Frozen HuBERT": "#9CA3AF", "Frozen HeAR": "#E5E7EB",
                  "Demo only (age + gender)": "#000000"}
    colors = [all_colors[k] for k in models]

    n_models = len(models)
    n_tasks = len(TASKS)
    x = np.arange(n_tasks)
    width = 0.8 / n_models
    offsets = np.linspace(-(n_models - 1) / 2 * width, (n_models - 1) / 2 * width, n_models)

    fig, ax = plt.subplots(figsize=(14, 6))

    for j, (name, color) in enumerate(zip(models.keys(), colors)):
        means, stds = [], []
        for t in TASKS:
            vals = [v for v in models[name][t] if v is not None]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.bar(x + offsets[j], means, width, yerr=stds, capsize=2,
               label=name, color=color, edgecolor="white", linewidth=0.5, zorder=3)

    ax.axhline(y=0.5, color="#D1D5DB", linestyle="--", linewidth=1, zorder=1)
    ax.set_ylabel("AUROC", fontsize=16, fontweight="medium")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=14)
    ax.set_ylim(0.3, 1.08)
    # Push legend below plot to avoid overlap with neurological bars (~1.0).
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4,
              fontsize=12, framealpha=0.95, edgecolor="#E5E7EB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(PAPER_DIR / "figS1_models.png", dpi=300, bbox_inches="tight")
    plt.savefig(PAPER_DIR / "figS1_models.pdf", bbox_inches="tight")
    print("Saved to paper/figS1_models.png")

    # Table
    print(f"\n{'Model':<25}", end="")
    for t in TASKS:
        print(f" {t:>14}", end="")
    print(f" {'MEAN':>10}")
    print("-" * 100)
    for name in models:
        print(f"{name:<25}", end="")
        cat_means = []
        for t in TASKS:
            vals = [v for v in models[name][t] if v is not None]
            m = np.mean(vals)
            cat_means.append(m)
            print(f" {m:>8.3f}±{np.std(vals):.3f}", end="")
        print(f" {np.mean(cat_means):>10.3f}")


if __name__ == "__main__":
    main()
