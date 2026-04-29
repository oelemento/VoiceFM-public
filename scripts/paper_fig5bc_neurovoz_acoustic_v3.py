#!/usr/bin/env python3
"""Figure 5b-c: NeuroVoz acoustic decomposition — VoiceFM-Whisper.

Panel b: Incremental R² — eGeMAPSv02 feature groups explaining P(PD).
Panel c: Cohen's d — PD vs Control effect sizes for top features.

Data source: results/neurovoz/gemaps_analysis.json (pre-computed eGeMAPSv02 analysis).

Format matches original paper_fig6_pd_combined.py style:
  - Horizontal bars (barh), invert_yaxis
  - YlOrRd colormap for R²
  - Feature-type colors + significance stars for Cohen's d

Usage:
    python scripts/paper_fig5bc_neurovoz_acoustic_v3.py
"""

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results_v3"
PAPER_DIR = PROJECT_ROOT / "paper_v3"
PAPER_DIR.mkdir(exist_ok=True)


# Feature-type color mapping (matches original fig6)
TYPE_COLORS = {
    "articulatory": "#c0392b",
    "spectral":     "#e67e22",
    "voicing":      "#8e44ad",
    "loudness":     "#2980b9",
    "phonatory":    "#95a5a6",
}


def classify_feature(name):
    n = name.lower()
    if any(x in n for x in ["f1amplitude", "f2amplitude", "f3amplitude",
                              "f1freq", "f2freq", "f3freq",
                              "f1frequency", "f2frequency", "f3frequency",
                              "f1bandwidth", "f2bandwidth", "f3bandwidth",
                              "bandwidth"]):
        return "articulatory"
    if any(x in n for x in ["spectral", "alpha", "hammar", "slope",
                              "flux", "spectralflux"]):
        return "spectral"
    if any(x in n for x in ["voiced", "voicing", "unvoiced",
                              "voicedseg", "meanvoiced", "stddevvoiced",
                              "mfcc", "logrel"]):
        return "voicing"
    if any(x in n for x in ["loudness", "intensity",
                              "loudnesspeaks", "equivalent"]):
        return "loudness"
    if any(x in n for x in ["f0", "jitter", "shimmer", "hnr", "pitch",
                              "f0semitone"]):
        return "phonatory"
    return "voicing"  # default for remaining eGeMAPSv02 features


def shorten_label(name, maxlen=32):
    # Make eGeMAPSv02 names more readable
    replacements = [
        ("From27.5Hz_sma3nz_", "_"),
        ("_sma3nz_", "_"),
        ("_sma3_", "_"),
        ("amean", "µ"),
        ("stddevNorm", "σ"),
        ("percentile20.0", "p20"),
        ("percentile50.0", "p50"),
        ("percentile80.0", "p80"),
        ("pctlrange0-2", "range"),
        ("meanRisingSlope", "↑slope"),
        ("stddevRisingSlope", "↑slopeσ"),
        ("meanFallingSlope", "↓slope"),
        ("stddevFallingSlope", "↓slopeσ"),
        ("LogRelF0", ""),
        ("amplitude", "amp"),
        ("frequency", "freq"),
    ]
    label = name
    for old, new in replacements:
        label = label.replace(old, new)
    if len(label) > maxlen:
        label = label[:maxlen - 1] + "…"
    return label


def main():
    with open(RESULTS / "neurovoz" / "gemaps_analysis.json") as f:
        gem = json.load(f)

    # ── Panel B data ──────────────────────────────────────────────────────────
    inc_data = gem["incremental"]
    # Skip "Other" group; show meaningful groups
    skip_groups = {"Other"}
    b_data = [(d["group"], d["r2"]) for d in inc_data if d["group"] not in skip_groups]
    gnames_b = [x[0] for x in b_data]
    r2_vals  = [x[1] for x in b_data]
    n_b = len(gnames_b)

    # ── Panel C data ──────────────────────────────────────────────────────────
    top20_raw = gem["group_diffs_top20"][:20]
    c_data = []
    for item in top20_raw:
        d, pv = item["cohens_d"], item["p_value"]
        feat_type = classify_feature(item["feature"])
        label = shorten_label(item["feature"])
        c_data.append((label, d, feat_type, pv))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(13, 6))

    # Panel B ─────────────────────────────────────────────────────────────────
    cmap = plt.cm.YlOrRd
    colors_b = [cmap(0.15 + 0.75 * i / max(n_b - 1, 1)) for i in range(n_b)]
    ax_b.barh(range(n_b), r2_vals, color=colors_b,
              edgecolor="white", linewidth=0.5, height=0.72)
    ax_b.set_yticks(range(n_b))
    ax_b.set_yticklabels(gnames_b, fontsize=10)
    ax_b.set_xlabel("R² explaining P(PD)", fontsize=11)
    ax_b.set_xlim(0, 1.15)
    ax_b.invert_yaxis()
    for i, v in enumerate(r2_vals):
        bold = "bold" if i == n_b - 1 else "normal"
        ax_b.text(v + 0.02, i, f".{round(v * 1000):03d}",
                  va="center", ha="left", fontsize=10, color="#333", fontweight=bold)
    if r2_vals:
        ax_b.text(r2_vals[-1] + 0.02, n_b - 1 + 0.38,
                  f"{int(round(r2_vals[-1] * 100))}% explained",
                  fontsize=9, color="#888", va="top", ha="left", fontstyle="italic")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.text(-0.02, 1.02, "b", transform=ax_b.transAxes,
              fontsize=21, fontweight="bold", va="bottom", ha="right")

    # Panel C ─────────────────────────────────────────────────────────────────
    feat_labels_c = [x[0] for x in c_data]
    d_vals_c      = [x[1] for x in c_data]
    feat_types_c  = [x[2] for x in c_data]
    p_vals_c      = [x[3] for x in c_data]
    bar_colors_c  = [TYPE_COLORS.get(t, "#888") for t in feat_types_c]

    ax_c.barh(range(len(feat_labels_c)), d_vals_c, color=bar_colors_c,
              edgecolor="white", linewidth=0.5, height=0.72)
    ax_c.set_yticks(range(len(feat_labels_c)))
    ax_c.set_yticklabels(feat_labels_c, fontsize=7.5)
    ax_c.set_xlabel("Cohen's d (PD − HC)", fontsize=11)
    ax_c.axvline(0, color="black", linewidth=0.5)
    ax_c.invert_yaxis()

    max_d = max(abs(v) for v in d_vals_c) if d_vals_c else 2.0
    ax_c.set_xlim(-max_d * 0.35, max_d * 1.35)

    for i, (dv, pv) in enumerate(zip(d_vals_c, p_vals_c)):
        if pv < 0.001:   stars = "***"
        elif pv < 0.01:  stars = "**"
        elif pv < 0.05:  stars = "*"
        else:            stars = "n.s."
        x_pos = dv + 0.05 if dv >= 0 else dv - 0.05
        ha = "left" if dv >= 0 else "right"
        color = "#333" if pv < 0.05 else "#aaa"
        ax_c.text(x_pos, i, stars, va="center", ha=ha, fontsize=9, color=color)

    legend_items = [mpatches.Patch(facecolor=c, label=t.capitalize())
                    for t, c in TYPE_COLORS.items()]
    ax_c.legend(handles=legend_items, fontsize=8, loc="lower right",
                frameon=True, framealpha=0.95, edgecolor="#ccc",
                handlelength=1.2, handleheight=1.0)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.text(-0.02, 1.02, "c", transform=ax_c.transAxes,
              fontsize=21, fontweight="bold", va="bottom", ha="right")

    plt.tight_layout()
    plt.savefig(PAPER_DIR / "fig5bc_neurovoz_acoustic.png", dpi=300, bbox_inches="tight")
    plt.savefig(PAPER_DIR / "fig5bc_neurovoz_acoustic.pdf", bbox_inches="tight")
    print("Saved to paper/fig5bc_neurovoz_acoustic.png")

    print(f"\nIncremental R²:")
    for gname, r2 in b_data:
        print(f"  {gname}: {r2:.3f}")
    print(f"\nTop 5 Cohen's d:")
    for label, d, t, p in c_data[:5]:
        print(f"  {label} ({t}): d={d:.3f}, p={p:.2e}")


if __name__ == "__main__":
    main()
