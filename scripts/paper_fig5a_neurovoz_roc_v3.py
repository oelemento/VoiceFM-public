#!/usr/bin/env python3
"""Figure 5a: NeuroVoz PD detection ROC curves — VoiceFM-Whisper.

ROC curves per task category (all, speech, DDK, vowel).
Uses roc_data_whisper_neurovoz.json from eval_whisper_neurovoz.py.

Usage:
    python scripts/paper_fig5a_neurovoz_roc_v3.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results_v3"
PAPER_DIR = PROJECT_ROOT / "paper_v3"
PAPER_DIR.mkdir(exist_ok=True)

# Colors matching original paper_fig6_pd_combined.py
TASK_COLORS = {
    "all": "#2196F3",
    "speech": "#4CAF50",
    "ddk": "#FF9800",
    "vowel": "#9C27B0",
}
TASK_LABELS = {
    "all": "All tasks",
    "speech": "Speech",
    "ddk": "DDK",
    "vowel": "Vowel",
}


def main():
    with open(RESULTS / "neurovoz" / "roc_data_whisper_neurovoz.json") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 6))

    for cat in ["all", "speech", "ddk", "vowel"]:
        if cat not in data:
            continue
        d = data[cat]
        fpr = d["fpr"]
        tpr = d["tpr"]
        auroc = d["auroc_mean"]
        std = d["auroc_std"]
        label = f"{TASK_LABELS[cat]} ({auroc:.3f}\u00b1{std:.3f})"
        ax.plot(fpr, tpr, color=TASK_COLORS[cat], linewidth=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal")
    ax.text(-0.02, 1.02, "a", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="right")

    plt.tight_layout()
    plt.savefig(PAPER_DIR / "fig5a_neurovoz_roc.png", dpi=300, bbox_inches="tight")
    plt.savefig(PAPER_DIR / "fig5a_neurovoz_roc.pdf", bbox_inches="tight")
    print("Saved to paper/fig5a_neurovoz_roc.png")

    for cat in ["all", "speech", "ddk", "vowel"]:
        if cat in data:
            d = data[cat]
            print(f"  {TASK_LABELS[cat]}: {d['auroc_mean']:.3f} \u00b1 {d['auroc_std']:.3f}")


if __name__ == "__main__":
    main()
