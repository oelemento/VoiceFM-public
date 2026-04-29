#!/usr/bin/env python3
"""Figure S2: Embedding interpretability — acoustic grounding.

Panel a: Ridge R² for VoiceFM-Whisper vs Frozen Whisper.
Panel b: Spearman ρ heatmap — acoustic features × VoiceFM-Whisper PCA components.

Usage:
    python scripts/paper_figS2_interpretability_v3.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results_v3"
PAPER_DIR = PROJECT_ROOT / "paper_v3"
PAPER_DIR.mkdir(exist_ok=True)

FEATURE_DISPLAY = {
    "f0_mean": "F0 (mean)", "f0_sd": "F0 (SD)", "f0_min": "F0 (min)",
    "f0_max": "F0 (max)", "f0_range": "F0 (range)",
    "jitter_local": "Jitter (local)", "jitter_rap": "Jitter (RAP)",
    "shimmer_local": "Shimmer (local)", "shimmer_apq3": "Shimmer (APQ3)",
    "hnr": "HNR", "cpps": "CPPS",
    "f1_mean": "F1 (mean)", "f2_mean": "F2 (mean)", "f3_mean": "F3 (mean)",
}


def main():
    with open(RESULTS / "voicefm_whisper_supplementary.json") as f:
        sup = json.load(f)

    ag_vw = sup["acoustic_grounding_vw"]
    ag_fw = sup["acoustic_grounding_fw"]
    pca_data = sup["pca"]

    # Sort by VoiceFM-Whisper R² descending
    features = sorted(ag_vw.keys(), key=lambda f: -ag_vw[f]["r2"])
    vw_r2 = [ag_vw[f]["r2"] for f in features]
    fw_r2 = [ag_fw[f]["r2"] for f in features]
    labels = [FEATURE_DISPLAY.get(f, f) for f in features]

    vw_mean = np.mean(vw_r2)
    fw_mean = np.mean(fw_r2)

    # ── Figure ────────────────────────────────────────────────────────
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 7),
                                      gridspec_kw={"width_ratios": [1, 0.8]})

    # Panel A: R² bars ─────────────────────────────────────────────────
    y = np.arange(len(features))
    h = 0.35
    ax_a.barh(y - h / 2, vw_r2, h, color="#2563EB", label="VoiceFM-Whisper", zorder=3)
    ax_a.barh(y + h / 2, fw_r2, h, color="#9CA3AF", label="Frozen Whisper", zorder=3)

    ax_a.axvline(0, color="black", linewidth=0.5)
    ax_a.axvline(vw_mean, color="#2563EB", linewidth=1, linestyle=":", alpha=0.5)
    ax_a.text(vw_mean + 0.01, len(features) - 0.5,
              f"mean={vw_mean:.2f}", fontsize=8, color="#2563EB", alpha=0.7)

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels, fontsize=10)
    ax_a.set_xlabel("Ridge regression R² (5-fold CV)", fontsize=11)
    ax_a.invert_yaxis()
    ax_a.legend(fontsize=9, loc="lower right")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.grid(axis="x", alpha=0.2)
    ax_a.text(-0.14, 1.02, "a", transform=ax_a.transAxes,
              fontsize=23, fontweight="bold", va="bottom")

    # Panel B: Spearman heatmap ────────────────────────────────────────
    hm = pca_data["heatmap"]
    ev = pca_data["explained_variance"]

    hm_features = [f for f in features if f in hm]
    hm_matrix = np.array([hm[f] for f in hm_features])
    hm_labels = [FEATURE_DISPLAY.get(f, f) for f in hm_features]
    pc_labels = [f"PC{i + 1}\n({ev[i]:.0%})" for i in range(len(ev))]

    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    im = ax_b.imshow(hm_matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    ax_b.set_xticks(range(len(pc_labels)))
    ax_b.set_xticklabels(pc_labels, fontsize=9)
    ax_b.set_yticks(range(len(hm_labels)))
    ax_b.set_yticklabels(hm_labels, fontsize=10)
    ax_b.set_xlabel("VoiceFM-Whisper PCA component", fontsize=10)
    fig.colorbar(im, ax=ax_b, label="Spearman ρ", shrink=0.8)
    ax_b.text(-0.22, 1.02, "b", transform=ax_b.transAxes,
              fontsize=23, fontweight="bold", va="bottom")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(PAPER_DIR / f"figS2_interpretability.{ext}",
                    dpi=300, bbox_inches="tight")
    print("Saved to paper/figS2_interpretability.png")
    print(f"\nVoiceFM-Whisper mean R²: {vw_mean:.3f}")
    print(f"Frozen Whisper mean R²: {fw_mean:.3f}")


if __name__ == "__main__":
    main()
