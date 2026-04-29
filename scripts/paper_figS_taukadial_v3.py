#!/usr/bin/env python3
"""Figure S (new): TAUKADIAL MCI detection — temporal MIL.

Two panels:
  a) Per-feature aggregation AUROC (mean / max / p90 / p95 / etc.)
  b) Comparison: baseline (mean-pool) vs temporal MIL (train-test, CV)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results_v3"
PAPER_DIR = PROJECT_ROOT / "paper_v3"
PAPER_DIR.mkdir(exist_ok=True)

BLUE = "#2563EB"
GRAY = "#9CA3AF"


def main():
    with open(RESULTS / "taukadial_temporal_mil.json") as f:
        d = json.load(f)

    mil = d["temporal_mil"]
    base = d["baseline"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5),
                                     gridspec_kw={"width_ratios": [1.5, 1]})

    # Panel a: per-feature aggregation
    feat_aurocs = mil["per_feature_aurocs"]
    order = ["mean", "max", "p90", "p95", "std", "frac_above_50",
             "frac_above_70", "top3_mean"]
    labels_disp = {
        "mean": "Mean",
        "max": "Max",
        "p90": "P90",
        "p95": "P95",
        "std": "Std",
        "frac_above_50": "Frac > 0.5",
        "frac_above_70": "Frac > 0.7",
        "top3_mean": "Top-3 mean",
    }
    feats = [f for f in order if f in feat_aurocs]
    vals = [feat_aurocs[f] for f in feats]

    colors = [BLUE if v >= 0.7 else GRAY for v in vals]
    bars = ax_a.bar(range(len(feats)), vals, color=colors, edgecolor="white")
    for i, (v, b) in enumerate(zip(vals, bars)):
        ax_a.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                  ha="center", fontsize=21, fontweight="bold" if v >= 0.7 else "normal",
                  color=BLUE if v >= 0.7 else "#6B7280")

    ax_a.set_xticks(range(len(feats)))
    ax_a.set_xticklabels([labels_disp[f] for f in feats], fontsize=9, rotation=30, ha="right")
    ax_a.set_ylabel("AUROC", fontsize=11)
    ax_a.set_ylim(0.4, 0.9)
    ax_a.axhline(0.5, color="#E5E7EB", linewidth=1, linestyle="--", zorder=0)
    ax_a.set_title("a   TAUKADIAL: per-feature aggregation AUROC",
                   fontsize=11, pad=8, loc="left")
    ax_a.grid(axis="y", alpha=0.15)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # Panel b: baseline vs MIL
    methods = ["Baseline\n(mean-pool)", "Temporal MIL"]
    tt = [base["train_test_auroc"], mil["train_test_auroc"]]
    cv = [base.get("cv_auroc", base.get("cv_auroc_mean", 0)), mil["cv_auroc_mean"]]

    x = np.arange(2)
    w = 0.35
    ax_b.bar(x - w / 2, tt, w, color=BLUE, label="Train→Test", edgecolor="white")
    ax_b.bar(x + w / 2, cv, w, color=GRAY, label="5-fold CV", edgecolor="white")

    for xi, (a, b) in enumerate(zip(tt, cv)):
        ax_b.text(xi - w / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=21,
                  fontweight="bold", color=BLUE)
        ax_b.text(xi + w / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=9,
                  color="#6B7280")

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(methods, fontsize=10)
    ax_b.set_ylabel("AUROC", fontsize=11)
    ax_b.set_ylim(0.4, 0.85)
    ax_b.axhline(0.5, color="#E5E7EB", linewidth=1, linestyle="--", zorder=0)
    ax_b.set_title("b   Baseline vs MIL", fontsize=11, pad=8, loc="left")
    ax_b.legend(loc="upper left", fontsize=9, frameon=False)
    ax_b.grid(axis="y", alpha=0.15)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    n_train = mil["n_train"]
    n_test = mil["n_test"]
    fig.suptitle(f"TAUKADIAL MCI detection (n_train={n_train}, n_test={n_test})",
                 fontsize=11, y=1.02)

    plt.tight_layout()
    plt.savefig(PAPER_DIR / "figS_taukadial.png", dpi=300, bbox_inches="tight")
    plt.savefig(PAPER_DIR / "figS_taukadial.pdf", bbox_inches="tight")
    print("Saved to paper_v3/figS_taukadial.png")
    print(f"\nMIL train→test: {mil['train_test_auroc']:.3f}")
    print(f"MIL 5-fold CV:  {mil['cv_auroc_mean']:.3f} ± {mil['cv_auroc_std']:.3f}")
    print(f"Baseline:       {base['train_test_auroc']:.3f}")


if __name__ == "__main__":
    main()
