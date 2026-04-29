#!/usr/bin/env python3
"""Figure 3b: Few-shot learning curves — VoiceFM-Whisper vs Frozen Whisper.

1×3 grid: one panel per dataset (MDVR-KCL, SVD, Coswara).
k=1,2,5,10,20 with mean ± std bands.

Usage:
    python scripts/paper_fig3b_fewshot_v3.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results_v3"
PAPER_DIR = PROJECT_ROOT / "paper_v3"
PAPER_DIR.mkdir(exist_ok=True)

C_VW = "#2196F3"
C_FW = "#9E9E9E"


def main():
    with open(RESULTS / "fewshot_results_whisper.json") as f:
        data = json.load(f)

    datasets = ["MDVR-KCL", "SVD", "Coswara"]
    titles = ["MDVR-KCL (Parkinson's)", "SVD (Voice Pathology)", "Coswara (COVID-19)"]
    ks = [1, 2, 5, 10, 20]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for idx, (ds, title) in enumerate(zip(datasets, titles)):
        ax = axes[idx]
        ds_data = data.get(ds, {})

        vw_means, vw_stds, fw_means, fw_stds = [], [], [], []
        valid_ks = []
        for k in ks:
            vw = ds_data.get(f"k={k}", {}).get("voicefm_whisper", {})
            fw = ds_data.get(f"k={k}", {}).get("frozen_whisper", {})
            vw_m = vw.get("mean_auroc")
            fw_m = fw.get("mean_auroc")
            if vw_m is None or fw_m is None or np.isnan(vw_m) or np.isnan(fw_m):
                continue
            valid_ks.append(k)
            vw_means.append(vw_m)
            vw_stds.append(vw.get("std_auroc", 0))
            fw_means.append(fw_m)
            fw_stds.append(fw.get("std_auroc", 0))

        vw_means = np.array(vw_means)
        vw_stds = np.array(vw_stds)
        fw_means = np.array(fw_means)
        fw_stds = np.array(fw_stds)

        ax.fill_between(valid_ks, vw_means - vw_stds, vw_means + vw_stds,
                         alpha=0.15, color=C_VW)
        ax.fill_between(valid_ks, fw_means - fw_stds, fw_means + fw_stds,
                         alpha=0.15, color=C_FW)
        ax.plot(valid_ks, vw_means, "o-", color=C_VW, linewidth=2, markersize=5,
                label="VoiceFM-Whisper")
        ax.plot(valid_ks, fw_means, "s--", color=C_FW, linewidth=2, markersize=5,
                label="Frozen Whisper")

        ax.axhline(y=0.5, color="black", linestyle=":", linewidth=0.6, alpha=0.3)
        ax.set_title(title, fontsize=23, fontweight="bold")
        ax.set_xlabel("k (samples per class)", fontsize=14)
        ax.set_ylabel("AUROC", fontsize=14)
        ax.set_xticks(valid_ks)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if idx == 0:
            ax.legend(fontsize=12, loc="lower right")

    fig.text(0.01, 0.98, "b", fontsize=23, fontweight="bold", va="top")
    plt.tight_layout(rect=[0.02, 0, 1, 1])
    plt.savefig(PAPER_DIR / "fig3b_fewshot.png", dpi=300, bbox_inches="tight")
    plt.savefig(PAPER_DIR / "fig3b_fewshot.pdf", bbox_inches="tight")
    print("Saved to paper/fig3b_fewshot.png")

    for ds in datasets:
        print(f"\n{ds}:")
        ds_data = data.get(ds, {})
        for k in ks:
            vw = ds_data.get(f"k={k}", {}).get("voicefm_whisper", {})
            fw = ds_data.get(f"k={k}", {}).get("frozen_whisper", {})
            vw_m = vw.get("mean_auroc")
            fw_m = fw.get("mean_auroc")
            vw_s = f"{vw_m:.3f}" if vw_m is not None else "  n/a"
            fw_s = f"{fw_m:.3f}" if fw_m is not None else "  n/a"
            print(f"  k={k:2d}: VW={vw_s}  FW={fw_s}")


if __name__ == "__main__":
    main()
