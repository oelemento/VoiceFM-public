#!/usr/bin/env python3
"""Figure 4: Recording task attribution — VoiceFM-Whisper.

Panel a: Per-recording-type AUROC heatmap (top tasks × 5 categories).
Panel b: Greedy forward selection curve.

Uses voicefm_whisper_full_eval.json (task_stratified + recording_attribution).

Usage:
    python scripts/paper_fig4_attribution_v3.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results_v3"
PAPER_DIR = PROJECT_ROOT / "paper_v3"
PAPER_DIR.mkdir(exist_ok=True)

CLF_LABELS = ["gsd_control", "cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]
LABEL_DISPLAY = {
    "gsd_control": "Control\nvs Disease",
    "cat_voice": "Voice",
    "cat_neuro": "Neuro",
    "cat_mood": "Mood",
    "cat_respiratory": "Resp",
}


def simplify_task_name(name):
    """Shorten recording type names for display."""
    name = name.replace("Harvard Sentences-", "Harvard ")
    name = name.replace("Maximum phonation time", "Max phonation")
    if len(name) > 35:
        name = name[:32] + "..."
    return name


def main():
    # ── Panel A: Task-stratified heatmap from v3 5-seed per_task_aurocs ──
    import glob
    seed_files = sorted(glob.glob(str(RESULTS / "recording_attribution_whisper_seed*.json")))
    task_aurocs = {}  # task_name -> {label: [vals across seeds]}
    LABEL_MAP = {"is_control": "gsd_control", "cat_voice": "cat_voice",
                 "cat_neuro": "cat_neuro", "cat_mood": "cat_mood",
                 "cat_respiratory": "cat_respiratory"}
    for sf in seed_files:
        with open(sf) as f:
            sd = json.load(f)
        for task_name, fields in sd.get("per_task_aurocs", {}).items():
            for raw_label, std_label in LABEL_MAP.items():
                if raw_label in fields and fields[raw_label] is not None:
                    task_aurocs.setdefault(task_name, {}).setdefault(std_label, []).append(fields[raw_label])
    task_aurocs = {t: {lbl: float(np.mean(v)) for lbl, v in d.items()} for t, d in task_aurocs.items()}

    # Filter to tasks with ≥3 labels available
    filtered_tasks = {t: v for t, v in task_aurocs.items() if len(v) >= 3}
    if not filtered_tasks:
        print("No task-stratified data found!")
        return

    # Sort by mean AUROC across labels (descending)
    task_mean = {t: np.mean(list(v.values())) for t, v in filtered_tasks.items()}
    sorted_tasks = sorted(task_mean, key=lambda t: -task_mean[t])

    # Take top 25 for readability
    top_tasks = sorted_tasks[:25]

    # Build heatmap matrix
    matrix = np.full((len(top_tasks), len(CLF_LABELS)), np.nan)
    for i, task in enumerate(top_tasks):
        for j, label in enumerate(CLF_LABELS):
            matrix[i, j] = filtered_tasks[task].get(label, np.nan)

    # ── Panel B: Greedy selection curve (5-seed mean ± std) ─────────────
    all_seed_aurocs = []
    all_seed_baselines = []
    for sf in seed_files:
        with open(sf) as f:
            sd = json.load(f)
        gs = sd["greedy_selection"]
        aurocs = [s.get("mean_auroc", s.get("auroc", 0)) for s in gs]
        all_seed_aurocs.append(aurocs)
        at = sd.get("all_types_aurocs", {})
        all_seed_baselines.append(np.mean(list(at.values())))

    # Trim to common length, compute mean/std
    min_len = min(len(a) for a in all_seed_aurocs)
    trimmed = np.array([a[:min_len] for a in all_seed_aurocs])
    mean_aurocs = trimmed.mean(axis=0)
    std_aurocs = trimmed.std(axis=0)
    steps = list(range(1, min_len + 1))
    baseline_mean = np.mean(all_seed_baselines)

    greedy = True  # flag for plotting

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 9),
                                      gridspec_kw={"width_ratios": [1.2, 1]})

    # Panel A: Heatmap
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=0.3, vcenter=0.5, vmax=1.0)
    im = ax_a.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax_a.set_xticks(range(len(CLF_LABELS)))
    ax_a.set_xticklabels([LABEL_DISPLAY[l] for l in CLF_LABELS], fontsize=12)
    ax_a.set_yticks(range(len(top_tasks)))
    ax_a.set_yticklabels([simplify_task_name(t) for t in top_tasks], fontsize=10)

    # Annotate cells
    for i in range(len(top_tasks)):
        for j in range(len(CLF_LABELS)):
            val = matrix[i, j]
            if np.isfinite(val):
                color = "white" if val < 0.5 or val > 0.9 else "black"
                ax_a.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax_a, shrink=0.6, pad=0.02)
    cbar.set_label("AUROC", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax_a.text(-0.02, 1.02, "a", transform=ax_a.transAxes,
              fontsize=26, fontweight="bold", va="bottom", ha="right")

    # Panel B: Greedy curve (5-seed mean with shaded std)
    if greedy:
        ax_b.plot(steps, mean_aurocs, "-", color="#2563EB", linewidth=2.5)
        ax_b.fill_between(steps, mean_aurocs - std_aurocs, mean_aurocs + std_aurocs,
                          color="#2563EB", alpha=0.15)

        # All-types baseline (5-seed mean)
        ax_b.axhline(y=baseline_mean, color="#DC2626", linestyle="--", linewidth=1.5,
                     label=f"All types ({baseline_mean:.3f})")

        # Find and annotate peak
        peak_idx = int(np.argmax(mean_aurocs))
        peak_step = steps[peak_idx]
        peak_val = mean_aurocs[peak_idx]
        ax_b.plot(peak_step, peak_val, "o", color="#2563EB", markersize=8, zorder=5)
        ax_b.annotate(f"Peak: {peak_val:.3f}\n({peak_step} types)",
                      xy=(peak_step, peak_val),
                      xytext=(peak_step + 5, peak_val - 0.015),
                      fontsize=12, arrowprops=dict(arrowstyle="->", color="gray"),
                      color="#2563EB", fontweight="bold")

        ax_b.set_xlabel("Number of recording types", fontsize=13)
        ax_b.set_ylabel("Mean AUROC (5 categories)", fontsize=13)
        ax_b.tick_params(axis="both", labelsize=11)
        n_seeds = len(seed_files) if seed_files else 1
        ax_b.legend(fontsize=11, loc="lower right",
                    title=f"{n_seeds}-seed mean ± SD", title_fontsize=10)
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)
        ax_b.grid(axis="y", alpha=0.3)

    ax_b.text(-0.02, 1.02, "b", transform=ax_b.transAxes,
              fontsize=26, fontweight="bold", va="bottom", ha="right")

    plt.tight_layout()

    plt.savefig(PAPER_DIR / "fig4_attribution.png", dpi=300, bbox_inches="tight")
    plt.savefig(PAPER_DIR / "fig4_attribution.pdf", bbox_inches="tight")
    print("Saved to paper/fig4_attribution.png")

    # Top 5 recording types
    print("\nTop 5 recording types (by mean AUROC):")
    for i, task in enumerate(top_tasks[:5]):
        print(f"  {i+1}. {task}: {task_mean[task]:.3f}")


if __name__ == "__main__":
    main()
