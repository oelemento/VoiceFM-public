#!/usr/bin/env python3
"""Figure 3a: External dataset transfer — VoiceFM-Whisper vs Frozen Whisper.

Bar chart: AUROC on 4 external datasets.
VoiceFM-Whisper (5-seed, 256d) vs Frozen Whisper.

Usage:
    python scripts/paper_fig3_transfer_v3.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results_v3"
PAPER_DIR = PROJECT_ROOT / "paper_v3"
PAPER_DIR.mkdir(exist_ok=True)

DATASETS = [
    ("mdvr_kcl", "MDVR-KCL\n(PD, n=73)"),
    ("svd", "SVD\n(Voice Path., n=2041)"),
    ("coswara", "Coswara\n(COVID, n=2098)"),
]


def main():
    vw_means, vw_stds = [], []
    fw_means, fw_stds = [], []

    for ds_key, _ in DATASETS:
        # VoiceFM-Whisper (5-seed)
        with open(RESULTS / f"eval_h28_{ds_key}.json") as f:
            h28 = json.load(f)
        vw_vals = [s["auroc"] for s in h28["h28_whisper_ft4_256d"]]
        vw_means.append(np.mean(vw_vals))
        vw_stds.append(np.std(vw_vals))

        # Frozen Whisper baseline
        with open(RESULTS / f"eval_h27_{ds_key}.json") as f:
            h27 = json.load(f)
        fw_val = h27["frozen_whisper_1280d"]["auroc"]
        fw_std = h27["frozen_whisper_1280d"].get("std", 0)
        fw_means.append(fw_val)
        fw_stds.append(fw_std)

    labels = [lab for _, lab in DATASETS]
    x = np.arange(len(DATASETS))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars_vw = ax.bar(x - width / 2, vw_means, width, yerr=vw_stds, capsize=4,
                     label="VoiceFM-Whisper", color="#2196F3",
                     edgecolor="white", linewidth=0.5, zorder=3)
    bars_fw = ax.bar(x + width / 2, fw_means, width, yerr=fw_stds, capsize=4,
                     label="Frozen Whisper", color="#9E9E9E",
                     edgecolor="white", linewidth=0.5, zorder=3)

    # P-value annotations (Welch's t-test: 5 VoiceFM-Whisper seeds vs 5 frozen Whisper folds)
    fw_perfold_path = RESULTS / "frozen_whisper_external_perfold.json"
    if fw_perfold_path.exists():
        with open(fw_perfold_path) as f:
            fw_perfold = json.load(f)
        for i, (ds_key, _) in enumerate(DATASETS):
            with open(RESULTS / f"eval_h28_{ds_key}.json") as f:
                h28 = json.load(f)
            vw_vals = [s["auroc"] for s in h28["h28_whisper_ft4_256d"]]
            fw_folds = fw_perfold[ds_key]["fold_aurocs"]
            _, p_val = stats.ttest_ind(vw_vals, fw_folds, equal_var=False)

            if p_val < 0.05:
                y_max = max(vw_means[i] + vw_stds[i], fw_means[i] + fw_stds[i])
                y_bar = y_max + 0.025
                x1 = i - width / 2
                x2 = i + width / 2
                ax.plot([x1, x1, x2, x2],
                        [y_bar - 0.008, y_bar, y_bar, y_bar - 0.008],
                        color="black", linewidth=1, zorder=4)
                label = f"p={p_val:.3f}" if p_val >= 0.001 else f"p={p_val:.1e}"
                ax.text(i, y_bar + 0.008, label, ha="center", va="bottom",
                        fontsize=7.5, color="black", zorder=4)
            print(f"  {ds_key}: Welch t-test p={p_val:.4f}")

    ax.axhline(y=0.5, color="#D1D5DB", linestyle="--", linewidth=1, zorder=1)
    ax.set_ylabel("AUROC", fontsize=12, fontweight="medium")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0.4, 1.08)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95, edgecolor="#E5E7EB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.text(-0.02, 1.02, "a", transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="bottom", ha="right")

    plt.tight_layout()

    plt.savefig(PAPER_DIR / "fig3_transfer.png", dpi=300, bbox_inches="tight")
    plt.savefig(PAPER_DIR / "fig3_transfer.pdf", bbox_inches="tight")
    print("Saved to paper/fig3_transfer.png")

    print(f"\n{'Dataset':<20} {'VoiceFM-Whisper':>18} {'Frozen Whisper':>16} {'Delta':>8}")
    print("-" * 65)
    for i, (ds_key, _) in enumerate(DATASETS):
        print(f"{ds_key:<20} {vw_means[i]:.3f}±{vw_stds[i]:.3f}     "
              f"{fw_means[i]:.3f}            {vw_means[i]-fw_means[i]:+.3f}")


if __name__ == "__main__":
    main()
