#!/usr/bin/env python3
"""Figure S (new): Prospective test — VoiceFM-Whisper held-out 138 participants.

Two panels:
  a) 5 disease categories (control, voice, neuro, mood, resp)
  b) 9 individual diagnoses (PD, AD/MCI, MTD, COPD/asthma, depression, airway,
     benign lesion, anxiety, laryngeal dystonia)
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

CATEGORIES = [
    ("gsd_control", "Control vs Disease"),
    ("cat_voice", "Voice"),
    ("cat_neuro", "Neuro"),
    ("cat_mood", "Mood"),
    ("cat_respiratory", "Respiratory"),
]

DIAGNOSES = [
    ("gsd_parkinsons", "Parkinson's disease"),
    ("gsd_alz_dementia_mci", "Alzheimer's/dementia/MCI"),
    ("gsd_mtd", "Muscle tension dysphonia"),
    ("gsd_copd_asthma", "COPD/Asthma"),
    ("gsd_depression", "Depression/MDD"),
    ("gsd_airway_stenosis", "Airway stenosis"),
    ("gsd_benign_lesion", "Benign vocal lesion"),
    ("gsd_anxiety", "Anxiety disorder"),
    ("gsd_laryngeal_dystonia", "Laryngeal dystonia"),
]


def collect(d, items):
    vw_means, vw_stds, vh_means, vh_stds = [], [], [], []
    for key, _ in items:
        vw_vals = d.get(f"voicefm_whisper/{key}", [])
        vh_vals = d.get(f"voicefm_hubert/{key}", [])
        vw_means.append(np.mean(vw_vals) if vw_vals else 0)
        vw_stds.append(np.std(vw_vals) if vw_vals else 0)
        vh_means.append(np.mean(vh_vals) if vh_vals else 0)
        vh_stds.append(np.std(vh_vals) if vh_vals else 0)
    return vw_means, vw_stds, vh_means, vh_stds


def plot_panel(ax, items, vw_m, vw_s, vh_m, vh_s, panel_label, title=None):
    x = np.arange(len(items))
    w = 0.38
    ax.bar(x - w / 2, vw_m, w, yerr=vw_s, color=BLUE, label="VoiceFM-Whisper",
           capsize=3, edgecolor="white", error_kw={"linewidth": 1.0})
    ax.bar(x + w / 2, vh_m, w, yerr=vh_s, color=GRAY, label="VoiceFM-HuBERT",
           capsize=3, edgecolor="white", error_kw={"linewidth": 1.0})

    for xi, (mvw, mvh) in enumerate(zip(vw_m, vh_m)):
        ax.text(xi - w / 2, mvw + 0.025, f"{mvw:.3f}", ha="center", fontsize=18,
                fontweight="bold", color=BLUE)
        ax.text(xi + w / 2, mvh + 0.025, f"{mvh:.3f}", ha="center", fontsize=7.5,
                color="#6B7280")

    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in items], fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("AUROC (3-seed mean ± SD)", fontsize=10)
    ax.set_ylim(0.5, 1.08)
    ax.axhline(0.5, color="#E5E7EB", linewidth=1, linestyle="--", zorder=0)
    if title:
        ax.set_title(title, fontsize=11, pad=6)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="bottom", ha="right")


def main():
    with open(RESULTS / "prospective_test_probes.json") as f:
        d = json.load(f)

    vw_m_c, vw_s_c, vh_m_c, vh_s_c = collect(d, CATEGORIES)
    vw_m_d, vw_s_d, vh_m_d, vh_s_d = collect(d, DIAGNOSES)

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(11, 9),
                                     gridspec_kw={"height_ratios": [1, 1.15]})

    plot_panel(ax_a, CATEGORIES, vw_m_c, vw_s_c, vh_m_c, vh_s_c, "a",
               title="Disease categories")
    plot_panel(ax_b, DIAGNOSES, vw_m_d, vw_s_d, vh_m_d, vh_s_d, "b",
               title="Individual diagnoses")

    fig.suptitle("Prospective evaluation on held-out v3 cohort (n = 138 participants)",
                 fontsize=12, y=1.0)

    plt.tight_layout()
    plt.savefig(PAPER_DIR / "figS_prospective.png", dpi=300, bbox_inches="tight")
    plt.savefig(PAPER_DIR / "figS_prospective.pdf", bbox_inches="tight")
    print("Saved to paper_v3/figS_prospective.png")

    print("\nPer-diagnosis VoiceFM-Whisper AUROCs:")
    for (k, lbl), m, s in zip(DIAGNOSES, vw_m_d, vw_s_d):
        print(f"  {lbl}: {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    main()
