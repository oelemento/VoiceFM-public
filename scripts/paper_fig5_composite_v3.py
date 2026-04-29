#!/usr/bin/env python3
"""Figure 5: Application to Parkinson's Disease — composite 7-panel figure.

Row 1 (NeuroVoz): (a) ROC, (b) incremental R², (c) Cohen's d
Row 2 (mPower):   (d) ROC, (e) P(PD) trajectories, (f) incremental R²
Row 3:            (g) Cohen's d (mPower)

All panels pre-rendered at identical size/padding for consistent alignment.

Usage:
    python scripts/paper_fig5_composite_v3.py
"""

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results_v3"
PAPER_DIR = PROJECT_ROOT / "paper_v3"
PAPER_DIR.mkdir(exist_ok=True)

FS_TITLE = 34
FS_LABEL = 35
FS_AXIS = 34
FS_TICK = 30
FS_NOTE = 30


def panel_label(ax, letter):
    # Cached-panel pattern: each panel is rendered at PW × PH then pasted
    # into the composite cells. Panel letter sits just above the axes box,
    # at y=1.02 (axes-fraction) so it's inside the canvas's top margin.
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes,
            fontsize=50, fontweight="bold", va="bottom", ha="right")


# ── Panel A: NeuroVoz ROC ─────────────────────────────────────────────

def plot_a(ax):
    with open(RESULTS / "neurovoz" / "roc_data_whisper_neurovoz.json") as f:
        roc = json.load(f)
    styles = {
        "all":    {"color": "#2166AC", "lw": 2.5, "ls": "-"},
        "speech": {"color": "#B2182B", "lw": 2.0, "ls": "-"},
        "ddk":    {"color": "#F4A582", "lw": 1.5, "ls": "--"},
        "vowel":  {"color": "#92C5DE", "lw": 1.5, "ls": ":"},
    }
    labels = {"all": "All tasks", "speech": "Speech", "ddk": "DDK", "vowel": "Vowel"}
    for cat in ["all", "speech", "ddk", "vowel"]:
        if cat not in roc:
            continue
        d = roc[cat]
        s = styles[cat]
        auc_val = d.get("auroc", d.get("auroc_mean", 0))
        n_val = d.get("n_total", len(d.get("labels", [])))
        lbl = f"{labels[cat]} (AUC={auc_val:.2f})"
        ax.plot(d["fpr"], d["tpr"], color=s["color"], linewidth=s["lw"],
                linestyle=s["ls"], label=lbl)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("False positive rate", fontsize=FS_AXIS)
    ax.set_ylabel("True positive rate", fontsize=FS_AXIS)
    ax.set_title("VoiceFM-Whisper (frozen)", fontsize=FS_TITLE, fontweight="medium")
    ax.legend(fontsize=FS_TICK, loc="lower right", frameon=True, framealpha=0.95,
              edgecolor="#ccc", handlelength=1.8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    # Aspect not forced equal: keeps ROC frame size matching bar-chart panels.
    ax.tick_params(labelsize=FS_TICK)
    panel_label(ax, "a")


# ── Panel B: NeuroVoz incremental R² ──────────────────────────────────

def plot_b(ax):
    with open(RESULTS / "neurovoz" / "gemaps_analysis.json") as f:
        gem = json.load(f)
    # Collapse to 5 canonical cumulative groups matching manuscript narrative.
    # Data order in JSON: F0 → J/S → HNR → Loudness → Formants → Bandwidth
    #                   → Spectral → MFCC → Voicing → Other
    inc_by_group = {row["group"]: row["r2"] for row in gem["incremental"]}
    for required in ("HNR", "Loudness", "Formants", "Spectral", "Other"):
        assert required in inc_by_group, f"Missing group {required} in neurovoz/gemaps_analysis.json"
    r2_data = [
        ("Phonatory",   inc_by_group["HNR"]),       # F0 + J/S + HNR cumulative
        ("+ Loudness",  inc_by_group["Loudness"]),
        ("+ Formants",  inc_by_group["Formants"]),
        ("+ Spectral",  inc_by_group["Spectral"]),
        ("+ All features", inc_by_group["Other"]),  # full 94-feature cumulative
    ]
    labels = [d[0] for d in r2_data]
    vals = [d[1] for d in r2_data]
    cmap = plt.cm.YlOrRd
    colors = [cmap(0.15 + 0.75 * i / max(len(vals) - 1, 1)) for i in range(len(vals))]
    ax.barh(range(len(labels)), vals, color=colors,
            edgecolor="white", linewidth=0.5, height=0.72)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.set_xlabel("R² explaining P(PD)", fontsize=FS_AXIS)
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        bold = "bold" if i == len(vals) - 1 else "normal"
        ax.text(v + 0.015, i, f".{round(v * 1000):03d}", va="center", ha="left",
                fontsize=FS_NOTE, color="#333", fontweight=bold)
    ax.tick_params(labelsize=FS_TICK)
    panel_label(ax, "b")


# ── Panel C: NeuroVoz Cohen's d ───────────────────────────────────────

def plot_c(ax):
    with open(RESULTS / "neurovoz" / "gemaps_analysis.json") as f:
        gem = json.load(f)
    top20 = gem["group_diffs_top20"][:12]

    def classify(n):
        nl = n.lower()
        if any(x in nl for x in ["f1amp", "f2amp", "f3amp", "f1freq", "f2freq", "f3freq",
                                   "f1band", "f2band", "f3band", "bandwidth"]):
            return "articulatory"
        if any(x in nl for x in ["spectral", "slope", "flux"]):
            return "spectral"
        if any(x in nl for x in ["mfcc", "voiced", "voicing", "unvoiced"]):
            return "voicing"
        if any(x in nl for x in ["loudness"]):
            return "loudness"
        return "phonatory"

    type_colors = {"articulatory": "#c0392b", "spectral": "#e67e22",
                   "voicing": "#8e44ad", "loudness": "#2980b9", "phonatory": "#95a5a6"}

    # Shorten feature names for readability
    def shorten(name):
        return (name.replace("amplitudeLogRelF0_sma3nz_amean", " amp rel F0")
                    .replace("_sma3nz_stddevNorm", " σ")
                    .replace("_sma3nz_amean", " μ")
                    .replace("_sma3_amean", " μ")
                    .replace("_sma3_stddevNorm", " σ")
                    .replace("UnvoicedSegmentLength", "Unvoiced len")
                    .replace("VoicedSegmentsPerSec", "Voiced seg/sec")
                    .replace("From27.5Hz", "")
                    .replace("semitone", "semitone"))[:30]
    feat_labels = [shorten(d["feature"]) for d in top20]
    d_vals = [d["cohens_d"] for d in top20]
    p_vals = [d["p_value"] for d in top20]
    feat_types = [classify(d["feature"]) for d in top20]
    bar_colors = [type_colors[t] for t in feat_types]

    ax.barh(range(len(feat_labels)), d_vals, color=bar_colors,
            edgecolor="white", linewidth=0.5, height=0.72)
    ax.set_yticks(range(len(feat_labels)))
    ax.set_yticklabels(feat_labels, fontsize=FS_TICK - 4)
    ax.set_xlabel("Cohen's d (PD − HC)", fontsize=FS_AXIS)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    # Widen x range so bars and stars fit
    max_d = max(abs(v) for v in d_vals)
    ax.set_xlim(-2.0, 2.8)
    for i, (dv, pv) in enumerate(zip(d_vals, p_vals)):
        if pv < 0.001: stars = "***"
        elif pv < 0.01: stars = "**"
        elif pv < 0.05: stars = "*"
        else: stars = "n.s."
        x_pos = dv + 0.08 if dv >= 0 else dv - 0.08
        ha = "left" if dv >= 0 else "right"
        color = "#333" if pv < 0.05 else "#999"
        ax.text(x_pos, i, stars, va="center", ha=ha, fontsize=FS_NOTE, color=color)
    # Color-coded bars (Articulatory red, Spectral orange, Voicing purple,
    # Loudness blue, Phonatory gray) — also described in the figure caption.
    # Legend in lower-right corner: bars at the bottom of the panel are all
    # negative-going (extend left from x=0), so the lower-right quadrant is
    # bar-free and the legend doesn't overlap data. Smaller font keeps it
    # compact.
    legend_items = [mpatches.Patch(facecolor=col, label=cat.capitalize())
                    for cat, col in type_colors.items()]
    ax.legend(handles=legend_items, fontsize=FS_NOTE - 8,
              loc="lower right", frameon=True, framealpha=0.95,
              edgecolor="#ccc", handlelength=1.0, handleheight=0.9,
              labelspacing=0.3)
    ax.tick_params(labelsize=FS_TICK)
    panel_label(ax, "c")


# ── Panel D: mPower ROC ──────────────────────────────────────────────

def plot_d(ax):
    with open(RESULTS / "mpower_pd_whisper_posthoc.json") as f:
        posthoc = json.load(f)
    roc = posthoc["roc"]
    for key, color, lw, alpha, label_prefix in [
        ("sustained", "#2166AC", 2.5, 1.0, "Sustained"),
        ("countdown", "#B2182B", 1.5, 0.5, "Countdown"),
    ]:
        if key not in roc:
            continue
        d = roc[key]
        ax.plot(d["fpr"], d["tpr"], color=color, linewidth=lw, alpha=alpha,
                label=f"{label_prefix} (AUC = {d['auroc']:.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("False positive rate", fontsize=FS_AXIS)
    ax.set_ylabel("True positive rate", fontsize=FS_AXIS)
    ax.set_title("VoiceFM-Whisper (fine-tuned)", fontsize=FS_TITLE, fontweight="medium")
    ax.legend(fontsize=FS_TICK, loc="lower right", frameon=True, framealpha=0.95,
              edgecolor="#ccc", handlelength=1.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    # Aspect not forced equal: keeps ROC frame size matching bar-chart panels.
    ax.tick_params(labelsize=FS_TICK)
    panel_label(ax, "d")


# ── Panel E: mPower trajectories ─────────────────────────────────────

def plot_e(ax):
    candidates = [
        RESULTS / "mpower_pd_whisper" / "test_predictions_deidentified.csv",
        RESULTS / "mpower_pd_whisper" / "whisper-voicefm_full_seed43" / "test_predictions.csv",
    ]
    pred_path = next((p for p in candidates if p.exists()), None)
    if pred_path is None:
        ax.text(0.5, 0.5, "No trajectory data", ha="center", transform=ax.transAxes, fontsize=FS_AXIS)
        panel_label(ax, "e")
        return

    pred_df = pd.read_csv(pred_path)
    if "participant" in pred_df.columns and "participant_id" not in pred_df.columns:
        pred_df = pred_df.rename(columns={"participant": "participant_id"})
    sus = pred_df[pred_df["recording_type"] == "sustained"].copy()
    if "months_since_enroll" in sus.columns:
        sus["month_bin"] = sus["months_since_enroll"].astype(int)
    else:
        sus["date"] = pd.to_datetime(sus["created_on"], unit="ms", errors="coerce")
        sus = sus.dropna(subset=["date"])
        first_date = sus.groupby("participant_id")["date"].min()
        sus["months"] = sus.apply(
            lambda r: (r["date"] - first_date[r["participant_id"]]).days / 30.44, axis=1)
        sus["month_bin"] = np.floor(sus["months"]).astype(int)
    max_months = 5

    colors = {"PD": "#B2182B", "Control": "#2166AC"}

    # Individual trajectories
    rng = np.random.RandomState(42)
    indiv = sus[sus["month_bin"] <= max_months].groupby(
        ["participant_id", "is_pd", "month_bin"]).agg(mean_pd=("prob_pd", "mean")).reset_index()
    bin_counts = indiv.groupby("participant_id")["month_bin"].nunique()
    keep = bin_counts[bin_counts >= 2].index
    indiv = indiv[indiv["participant_id"].isin(keep)]

    for lbl, val in [("Control", 0), ("PD", 1)]:
        pids = indiv[indiv["is_pd"] == val]["participant_id"].unique()
        chosen = rng.choice(pids, min(50, len(pids)), replace=False)
        sub = indiv[indiv["participant_id"].isin(chosen)]
        for pid, grp in sub.groupby("participant_id"):
            grp_s = grp.sort_values("month_bin")
            ax.plot(grp_s["month_bin"], grp_s["mean_pd"],
                    color=colors[lbl], alpha=0.08, linewidth=0.5, zorder=1)

    # Group means — smoothed with rolling window
    from scipy.ndimage import uniform_filter1d
    for lbl, val in [("PD", 1), ("Control", 0)]:
        grp = sus[(sus["is_pd"] == val) & (sus["month_bin"] <= max_months)]
        binned = grp.groupby("month_bin").agg(
            mean_pd=("prob_pd", "mean"), n_parts=("participant_id", "nunique")).reset_index()
        binned = binned[binned["n_parts"] >= 5]
        if len(binned) >= 2:
            ax.plot(binned["month_bin"], binned["mean_pd"], color=colors[lbl],
                    linewidth=3.0, label=lbl, zorder=4)
        elif len(binned) == 1:
            ax.scatter(binned["month_bin"], binned["mean_pd"], color=colors[lbl],
                       s=80, zorder=4, label=lbl)

    ax.set_xlabel("Months since first recording", fontsize=FS_AXIS)
    ax.set_ylabel("P(PD)", fontsize=FS_AXIS)
    ax.set_xlim(0, max_months)
    ax.set_ylim(0, 1.0)
    # Legend top-right: PD trajectory hovers ~0.65, control ~0.20, so the
    # upper-right corner is empty space free of overlap.
    ax.legend(fontsize=FS_NOTE, loc="upper right", frameon=True, framealpha=0.95)
    ax.tick_params(labelsize=FS_TICK)
    panel_label(ax, "e")


# ── Panel F: mPower incremental R² ───────────────────────────────────

def plot_f(ax):
    with open(RESULTS / "mpower_gemaps" / "gemaps_mpower_analysis.json") as f:
        gem = json.load(f)
    # Match Panel B's 5-group structure for visual comparison
    inc_by_group = {row["group"]: row["r2"] for row in gem["incremental"]}
    for required in ("HNR", "Loudness", "Formants", "Spectral", "Other"):
        assert required in inc_by_group, f"Missing group {required} in mpower gemaps"
    r2_data = [
        ("Phonatory",      inc_by_group["HNR"]),       # F0 + J/S + HNR cumulative
        ("+ Loudness",     inc_by_group["Loudness"]),
        ("+ Formants",     inc_by_group["Formants"]),
        ("+ Spectral",     inc_by_group["Spectral"]),
        ("+ All features", inc_by_group["Other"]),     # full 94-feature cumulative
    ]
    labels = [d[0] for d in r2_data]
    vals = [d[1] for d in r2_data]
    cmap = plt.cm.YlOrRd
    colors = [cmap(0.15 + 0.75 * i / max(len(r2_data) - 1, 1)) for i in range(len(r2_data))]
    ax.barh(range(len(labels)), vals, color=colors,
            edgecolor="white", linewidth=0.5, height=0.72)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.set_xlabel("R² explaining P(PD)", fontsize=FS_AXIS)
    ax.set_xlim(0, 1.15)  # match panel b's xlim so frame size is identical
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        bold = "bold" if i == len(vals) - 1 else "normal"
        ax.text(v + 0.015, i, f".{int(v * 1000):03d}", va="center", ha="left",
                fontsize=FS_NOTE, color="#333", fontweight=bold)
    ax.tick_params(labelsize=FS_TICK)
    panel_label(ax, "f")


# ── Panel G: mPower Cohen's d ────────────────────────────────────────

def plot_g(ax):
    with open(RESULTS / "mpower_gemaps" / "gemaps_mpower_analysis.json") as f:
        gem = json.load(f)
    top20 = gem["group_diffs_top20"][:12]

    def classify(n):
        nl = n.lower()
        if any(x in nl for x in ["f1amp", "f2amp", "f3amp", "f1freq", "f2freq", "f3freq",
                                   "f1band", "f2band", "f3band", "bandwidth"]):
            return "articulatory"
        if any(x in nl for x in ["spectral", "slope", "flux"]):
            return "spectral"
        if any(x in nl for x in ["mfcc", "voiced", "voicing", "unvoiced"]):
            return "voicing"
        if any(x in nl for x in ["loudness"]):
            return "loudness"
        return "phonatory"

    type_colors = {"articulatory": "#c0392b", "spectral": "#e67e22",
                   "voicing": "#8e44ad", "loudness": "#2980b9", "phonatory": "#95a5a6"}

    label_map = {
        "F2bandwidth_sma3nz_stddevNorm": "F2 bandwidth SD",
        "F0semitoneFrom27.5Hz_sma3nz_amean": "F0 mean",
        "F0semitoneFrom27.5Hz_sma3nz_percentile50.0": "F0 median",
        "F0semitoneFrom27.5Hz_sma3nz_percentile20.0": "F0 p20",
        "F0semitoneFrom27.5Hz_sma3nz_percentile80.0": "F0 p80",
        "F1bandwidth_sma3nz_stddevNorm": "F1 bandwidth SD",
        "F1bandwidth_sma3nz_amean": "F1 bandwidth",
        "F3frequency_sma3nz_amean": "F3 frequency",
        "F2frequency_sma3nz_amean": "F2 frequency",
        "F1frequency_sma3nz_amean": "F1 frequency",
        "loudnessPeaksPerSec": "Loudness peaks/sec",
        "mfcc1_sma3_amean": "MFCC1 mean",
        "F3bandwidth_sma3nz_stddevNorm": "F3 bandwidth SD",
        "F3bandwidth_sma3nz_amean": "F3 bandwidth",
        "F2amplitudeLogRelF0_sma3nz_amean": "F2 amp. rel. F0",
        "F3amplitudeLogRelF0_sma3nz_amean": "F3 amp. rel. F0",
        "shimmerLocaldB_sma3nz_amean": "Shimmer",
        "jitterLocal_sma3nz_amean": "Jitter",
        "HNRdBACF_sma3nz_amean": "HNR",
    }

    def shorten(name):
        return label_map.get(name, name.split("_")[0][:20])

    # Mark sex-confounded features
    sex_confounded = {"F0 mean", "F0 median", "F0 p20", "F0 p80"}

    feat_labels = [shorten(d["feature"]) for d in top20]
    d_vals = [d["cohens_d"] for d in top20]
    p_vals = [d["p_value"] for d in top20]
    feat_types = [classify(d["feature"]) for d in top20]
    bar_colors = [type_colors[t] for t in feat_types]

    ax.barh(range(len(feat_labels)), d_vals, color=bar_colors,
            edgecolor="white", linewidth=0.5, height=0.72)
    ax.set_yticks(range(len(feat_labels)))
    # Add dagger to sex-confounded labels
    display_labels = [f"{l} \u2020" if l in sex_confounded else l for l in feat_labels]
    ax.set_yticklabels(display_labels, fontsize=FS_TICK - 2)
    ax.set_xlabel("Cohen's d (PD \u2212 HC)", fontsize=FS_AXIS)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, 0.7)
    for i, (dv, pv) in enumerate(zip(d_vals, p_vals)):
        if pv < 0.001: stars = "***"
        elif pv < 0.01: stars = "**"
        elif pv < 0.05: stars = "*"
        else: stars = "n.s."
        x_pos = dv + 0.02 if dv >= 0 else dv - 0.02
        ha = "left" if dv >= 0 else "right"
        color = "#333" if pv < 0.05 else "#999"
        ax.text(x_pos, i, stars, va="center", ha=ha, fontsize=FS_NOTE, color=color)
    # Dagger footnote
    ax.tick_params(labelsize=FS_TICK)
    panel_label(ax, "g")


# ── Main ─────────────────────────────────────────────────────────────

# ── Panel rendering (copied from old v1 paper_fig6_pd_combined.py) ───

PANEL_STYLE = {
    "font.family": "Arial",
    "font.size": 35,
    "axes.linewidth": 3.0,
    "xtick.major.width": 3.0,
    "ytick.major.width": 3.0,
    "xtick.labelsize": 35,
    "ytick.labelsize": 35,
    "axes.labelsize": 40,
}

# Deterministic layout: every panel renders into the SAME 10×10" canvas with
# a 6.5×6.5" axes box at the SAME position [x=2.6, y=1.2, w=6.5, h=6.5].
# This is more reliable than fighting `subplots_adjust` + `set_box_aspect` +
# `bbox_inches="tight"` — by placing the axes explicitly we know exactly
# where the box ends up in the saved image, so all panels' boxes align
# perfectly when pasted into the composite cells.
# All panels share the same canvas size and the same plot-frame HEIGHT, but
# ROC panels are square while bar panels fill the full width (more horizontal
# room for the bars). The user's rule: same height across panels; widths can
# differ.
# EVERY panel has a square plot box of the same size (BOX_SIZE × BOX_SIZE)
# matching panel g's width. Each panel's CANVAS width depends on its column —
# columns with longer y-tick labels get a wider canvas so the box sits at the
# same pixel y-range and same in-cell pixel x-range across rows. Composite
# uses width_ratios to size each column's cell to its panel's canvas, so
# panels paste flush and BOXES ALIGN vertically across rows.
BOX_SIZE_IN   = 8.5                  # plot box width = height (uniform)
BOX_BOTTOM_IN = 1.00
BOX_TOP_IN    = 0.85
BOX_RIGHT_IN  = 0.2

# Per-COLUMN left margin: same for every panel in that column so boxes align.
COLUMN = {"a": 0, "b": 1, "c": 2,
          "d": 0, "e": 1, "f": 2,
          "g": 0}
COLUMN_LEFT_IN = {
    0: 3.3,   # col 0 = a, d, g — widest label is g's "F2 bandwidth SD" (15 chars)
    1: 3.0,   # col 1 = b, e — b's "+ All features" (~14 chars)
    2: 4.5,   # col 2 = c, f — c's "StddevUnvoiced len σ" (~20 chars)
}
COLUMN_CANVAS_W = {col: COLUMN_LEFT_IN[col] + BOX_SIZE_IN + BOX_RIGHT_IN
                   for col in COLUMN_LEFT_IN}
PH_RENDER = BOX_BOTTOM_IN + BOX_SIZE_IN + BOX_TOP_IN          # 10.35"


def axes_rect_for(panel_id: str):
    """Square box of size BOX_SIZE positioned at the column's left margin.
    Canvas width depends on column; canvas height is uniform across panels.
    """
    col = COLUMN[panel_id]
    left = COLUMN_LEFT_IN[col]
    canvas_w = COLUMN_CANVAS_W[col]
    return [left / canvas_w,
            BOX_BOTTOM_IN / PH_RENDER,
            BOX_SIZE_IN / canvas_w,
            BOX_SIZE_IN / PH_RENDER]


def _render_panel(panel_id, plot_fn):
    cache_dir = PAPER_DIR / "_panel_cache"
    cache_dir.mkdir(exist_ok=True)
    out = cache_dir / f"fig5_{panel_id}.png"
    rect = axes_rect_for(panel_id)
    canvas_w = COLUMN_CANVAS_W[COLUMN[panel_id]]
    with plt.rc_context(PANEL_STYLE):
        fig = plt.figure(figsize=(canvas_w, PH_RENDER))
        ax = fig.add_axes(rect)
        plot_fn(ax)
        fig.savefig(out, dpi=200, facecolor="white")
        plt.close(fig)
    return out


def main():
    import matplotlib.image as mpimg

    panel_fns = {
        "a": plot_a, "b": plot_b, "c": plot_c,
        "d": plot_d, "e": plot_e, "f": plot_f, "g": plot_g,
    }

    print("Rendering panels...")
    panel_paths = {}
    for pid, fn in panel_fns.items():
        panel_paths[pid] = _render_panel(pid, fn)
        print(f"  {pid}: done")

    imgs = {k: mpimg.imread(str(v)) for k, v in panel_paths.items()}

    # Letter-page friendly aspect: figsize (20, 24) ≈ 8.5×11 ratio so the
    # composite scales to fill the available height on a 8.5×11 PDF page.
    # Row headers were dropped (caption now carries dataset descriptions),
    # so top_gap can be tight.
    top_gap = 0.005
    inter_gap = 0.020
    bot_gap = 0.005
    row_frac = (1.0 - top_gap - inter_gap - bot_gap) / 3.0

    top1 = 1.0 - top_gap
    bot1 = top1 - row_frac
    top2 = bot1 - inter_gap
    bot2 = top2 - row_frac
    top3 = bot2 - bot_gap
    bot3 = top3 - row_frac

    # Composite uses width_ratios = column canvas widths so each cell scales
    # by exactly the same factor (cell_w / canvas_w_for_that_column). This
    # makes every panel's plot box render at the same pixel size and lines up
    # vertically across rows.
    width_ratios = [COLUMN_CANVAS_W[c] for c in (0, 1, 2)]
    sum_w = sum(width_ratios)
    avg_canvas_w = sum_w / 3.0
    cell_w_avg = 20.0 / 3.0
    cell_h = cell_w_avg * (PH_RENDER / avg_canvas_w)
    composite_h = 3.0 * cell_h / (1.0 - top_gap - inter_gap - bot_gap)
    fig = plt.figure(figsize=(20, composite_h))
    LMARGIN, RMARGIN, WSPACE = 0.005, 0.995, 0.02

    gs1 = gridspec.GridSpec(1, 3, figure=fig, width_ratios=width_ratios,
                            left=LMARGIN, right=RMARGIN, top=top1, bottom=bot1,
                            wspace=WSPACE)
    gs2 = gridspec.GridSpec(1, 3, figure=fig, width_ratios=width_ratios,
                            left=LMARGIN, right=RMARGIN, top=top2, bottom=bot2,
                            wspace=WSPACE)
    gs3 = gridspec.GridSpec(1, 3, figure=fig, width_ratios=width_ratios,
                            left=LMARGIN, right=RMARGIN, top=top3, bottom=bot3,
                            wspace=WSPACE)

    def paste(gs_spec, img):
        ax = fig.add_subplot(gs_spec)
        ax.imshow(img)
        ax.axis("off")

    paste(gs1[0, 0], imgs["a"])
    paste(gs1[0, 1], imgs["b"])
    paste(gs1[0, 2], imgs["c"])
    paste(gs2[0, 0], imgs["d"])
    paste(gs2[0, 1], imgs["e"])
    paste(gs2[0, 2], imgs["f"])
    paste(gs3[0, 0], imgs["g"])

    out = PAPER_DIR / "fig5_pd_combined.png"
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
