#!/usr/bin/env python3
"""Figure S3: Embedding structure.

Panel a: NN retrieval — 2 sub-panels (category match, severity diff)
         VoiceFM-Whisper vs Frozen Whisper
Panel b: Within-participant consistency for VoiceFM-Whisper
         (intra-person vs inter-person cosine similarity)

Uses voicefm_whisper_supplementary.json (with within_participant key).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results_v3"
PAPER_DIR = PROJECT_ROOT / "paper_v3"
PAPER_DIR.mkdir(exist_ok=True)

BLUE = "#2563EB"
GRAY = "#9CA3AF"


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    with open(RESULTS / "voicefm_whisper_supplementary.json") as f:
        sup = json.load(f)

    nn_vw = sup["nn_vw"]
    nn_fw = sup["nn_fw"]
    wp = sup["within_participant"]


    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig,
                           left=0.08, right=0.96,
                           top=0.92, bottom=0.14,
                           wspace=0.30,
                           width_ratios=[1.0, 1.0])

    # ── Panel A: Diagnosis category match (NN retrieval) ──────────────
    ax = fig.add_subplot(gs[0, 0])
    vfm = nn_vw["cat_match_rate"]
    fw  = nn_fw["cat_match_rate"]
    ylim = (0, 0.75)
    x = np.array([0, 1])
    bars = ax.bar(x, [vfm, fw], 0.55, color=[BLUE, GRAY], alpha=0.9)
    for bar, val in zip(bars, [vfm, fw]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ylim[1] - ylim[0]) * 0.03,
                f"{val:.3f}", ha="center", fontsize=13,
                fontweight="bold" if bar == bars[0] else "normal",
                color=BLUE if bar == bars[0] else "#6B7280")
    ax.set_xticks(x)
    ax.set_xticklabels(["VoiceFM-Whisper", "Frozen Whisper"], fontsize=11)
    ax.set_ylim(ylim)
    ax.set_title("Diagnosis category match", fontsize=13, pad=6)
    ax.set_ylabel("k=5 nearest-neighbor match rate", fontsize=12)
    ax.grid(axis="y", alpha=0.15, linewidth=0.7)
    clean_ax(ax)
    ax.text(-0.18, 1.02, "a", transform=ax.transAxes,
            fontsize=22, fontweight="bold", va="bottom")

    # ── Panel B: VoiceFM-Whisper within-participant consistency ───────
    ax_b = fig.add_subplot(gs[0, 1])

    intra = wp["intra_mean"]
    inter = wp["inter_mean"]
    sep = wp["separation"]

    x = np.array([0, 1])
    ax_b.bar([0], [intra], 0.55, color=BLUE, alpha=0.9, edgecolor="white")
    ax_b.bar([1], [inter], 0.55, color=BLUE, alpha=0.4, edgecolor="white")

    for xi, val in [(0, intra), (1, inter)]:
        ax_b.text(xi, val + 0.005, f"{val:.3f}", ha="center", va="bottom",
                  fontsize=13, fontweight="bold", color=BLUE)

    ax_b.annotate("", xy=(1.35, inter), xytext=(1.35, intra),
                  arrowprops=dict(arrowstyle="<->", color="#374151", lw=1.5))
    ax_b.text(1.42, (intra + inter) / 2, f"Δ = {sep:.3f}",
              fontsize=13, va="center", fontweight="bold", color=BLUE)

    ax_b.set_xlim(-0.5, 1.9)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["Same\nperson", "Different\nperson"], fontsize=13)
    ax_b.set_ylabel("Mean cosine similarity", fontsize=14)
    y_lo = max(0.0, min(intra, inter) - 0.02)
    ax_b.set_ylim(y_lo, max(intra, inter) + 0.05)
    ax_b.set_title("VoiceFM-Whisper within-participant consistency",
                   fontsize=13, pad=10)
    ax_b.grid(axis="y", alpha=0.15, linewidth=0.7)

    clean_ax(ax_b)
    ax_b.text(-0.10, 1.02, "b", transform=ax_b.transAxes,
              fontsize=22, fontweight="bold", va="bottom")

    plt.savefig(PAPER_DIR / "figS3_embedding_structure.png", dpi=300, bbox_inches="tight")
    plt.savefig(PAPER_DIR / "figS3_embedding_structure.pdf", bbox_inches="tight")
    print("Saved to paper_v3/figS3_embedding_structure.png")

    print(f"\nNN match: VW={nn_vw['cat_match_rate']:.3f}, FW={nn_fw['cat_match_rate']:.3f}")
    print(f"Within-participant: intra={intra:.3f}, inter={inter:.3f}, Δ={sep:.3f}")


if __name__ == "__main__":
    main()
