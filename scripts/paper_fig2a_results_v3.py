#!/usr/bin/env python3
"""Figure 2a: GSD category AUROC comparison — 4 models.

Grouped bar chart: VoiceFM-Whisper, VoiceFM-HuBERT, Frozen Whisper, Frozen HuBERT.
All use corrected eval: all-recordings mean-pool, gsd_control label.

Left group: Average AUROC (mean across 5 tasks) with significance brackets.
Right groups: Individual task AUROCs.

Usage:
    python scripts/paper_fig2a_results_v3.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results_v3"

TASKS = ["gsd_control", "cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]
LABELS = ["Control\nvs Disease", "Voice\nDisorder", "Neuro-\nlogical", "Mood\nDisorder", "Respiratory\nDisorder"]


def load_models():
    """Load results from unified_gsd_probes.json (all 4 models, identical methodology)."""
    models = {}

    unified_path = RESULTS / "unified_gsd_probes.json"
    if not unified_path.exists():
        raise FileNotFoundError(f"Missing {unified_path}")

    with open(unified_path) as f:
        data = json.load(f)

    name_map = {
        "VoiceFM-Whisper": "voicefm_whisper",
        "VoiceFM-HuBERT": "voicefm_hubert",
        "Frozen Whisper": "frozen_whisper",
        "Frozen HuBERT": "frozen_hubert",
    }
    for display_name, prefix in name_map.items():
        key = f"{prefix}/gsd_control"
        if key in data:
            models[display_name] = {t: data[f"{prefix}/{t}"] for t in TASKS}

    return models


def compute_avg_auroc(models):
    """Compute per-fold average AUROC across tasks for each model."""
    avg = {}
    for name, task_data in models.items():
        vals = np.array([task_data[t] for t in TASKS])  # shape: (5_tasks, n_folds)
        fold_means = vals.mean(axis=0)  # mean across tasks for each fold
        avg[name] = fold_means
    return avg


def add_bracket(ax, x1, x2, y, p, h=0.008, fontsize=7.5):
    """Add significance bracket between x1 and x2 at height y."""
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        return  # don't draw
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=0.8)
    ax.text((x1 + x2) / 2, y + h + 0.002, stars, ha="center", va="bottom",
            fontsize=fontsize, color="black")


def main():
    models = load_models()
    model_names = ["VoiceFM-Whisper", "VoiceFM-HuBERT", "Frozen Whisper", "Frozen HuBERT"]
    colors = ["#2563EB", "#7C3AED", "#D97706", "#9CA3AF"]

    available = [(n, c) for n, c in zip(model_names, colors) if n in models]
    print(f"Models available: {[n for n, _ in available]}")

    n_models = len(available)
    avg_auroc = compute_avg_auroc(models)

    # X positions: "Average" at 0, then tasks at 1.5, 2.5, ...
    all_labels = ["Average\nAUROC"] + LABELS
    n_groups = len(all_labels)
    x = np.array([0] + [i + 1.5 for i in range(len(TASKS))])
    width = 0.8 / n_models
    offsets = np.linspace(-(n_models - 1) / 2 * width, (n_models - 1) / 2 * width, n_models)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for j, (name, color) in enumerate(available):
        means, stds = [], []

        # Average AUROC (first group)
        avg_vals = avg_auroc[name]
        means.append(np.mean(avg_vals))
        stds.append(np.std(avg_vals) if len(avg_vals) > 1 else 0)

        # Per-task AUROCs
        for t in TASKS:
            vals = [v for v in models[name][t] if v is not None]
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0)

        ax.bar(x + offsets[j], means, width, yerr=stds, capsize=3,
               label=name, color=color, edgecolor="white", linewidth=0.5, zorder=3)

    # ── Significance brackets on Average group ────────────────────────
    # Compare VoiceFM-Whisper (primary model) against each other model.
    bracket_pairs = []
    model_idx = {name: j for j, (name, _) in enumerate(available)}

    if "VoiceFM-Whisper" in avg_auroc:
        vw_vals = avg_auroc["VoiceFM-Whisper"]
        for other in ["VoiceFM-HuBERT", "Frozen Whisper", "Frozen HuBERT"]:
            if other not in avg_auroc:
                continue
            ov = avg_auroc[other]
            if len(vw_vals) == 1 and len(ov) > 1:
                _, p = stats.ttest_1samp(ov, vw_vals[0])
            elif len(ov) == 1 and len(vw_vals) > 1:
                _, p = stats.ttest_1samp(vw_vals, ov[0])
            else:
                _, p = stats.ttest_ind(vw_vals, ov, equal_var=False)
            bracket_pairs.append(("VoiceFM-Whisper", other, p))

    # Draw brackets (only significant ones, stacked, fit beneath ylim=1.0)
    sig_pairs = [(n1, n2, p) for n1, n2, p in bracket_pairs if p < 0.05]
    sig_pairs.sort(key=lambda t: abs(model_idx[t[0]] - model_idx[t[1]]))  # innermost first
    n_sig = len(sig_pairs)
    y_base = 0.965
    y_top = 0.99
    step = (y_top - y_base) / max(n_sig - 1, 1) if n_sig > 1 else 0
    for k, (n1, n2, p) in enumerate(sig_pairs):
        x1 = 0 + offsets[model_idx[n1]]
        x2 = 0 + offsets[model_idx[n2]]
        y = y_base + k * step
        add_bracket(ax, x1, x2, y, p)

    # ── Separator line between Average and per-task groups ────────────
    ax.axvline(x=0.75, color="#E5E7EB", linestyle="-", linewidth=1, zorder=1)

    ax.axhline(y=0.5, color="#D1D5DB", linestyle="--", linewidth=1, zorder=1)
    ax.set_ylabel("AUROC", fontsize=12, fontweight="medium")
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=10)
    ax.set_ylim(0.7, 1.0)
    ax.set_yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95, edgecolor="#E5E7EB", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.text(-0.02, 1.02, "a", transform=ax.transAxes,
            fontsize=20, fontweight="bold", va="bottom", ha="right")

    plt.tight_layout()

    out_dir = PROJECT_ROOT / "paper_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "fig2a_results.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "fig2a_results.pdf", bbox_inches="tight")
    print("Saved to paper/fig2a_results.png")

    # Print summary
    print(f"\n{'Model':<20}  {'Average':>8}  ", end="")
    for t in TASKS:
        print(f"  {t:>12}", end="")
    print()
    print("-" * 100)
    for name, _ in available:
        avg_m = np.mean(avg_auroc[name])
        avg_s = np.std(avg_auroc[name]) if len(avg_auroc[name]) > 1 else 0
        print(f"{name:<20}  {avg_m:>5.3f}±{avg_s:.3f}", end="  ")
        for t in TASKS:
            vals = [v for v in models[name][t] if v is not None]
            print(f"  {np.mean(vals):>8.3f}±{np.std(vals):.3f}", end="")
        print()

    print(f"\nSignificance (Average AUROC):")
    for n1, n2, p in bracket_pairs:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {n1} vs {n2}: p={p:.4f} {sig}")


if __name__ == "__main__":
    main()
