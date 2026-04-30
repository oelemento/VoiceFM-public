#!/usr/bin/env python3
"""Figure 3: Prospective evaluation on held-out v3 cohort (n=138 participants).

Panel a: Bar chart of 5 disease categories (VoiceFM-Whisper vs VoiceFM-HuBERT)
Panel b: Table of 9 individual diagnoses with AUROCs

Same-row 2-panel layout, blue/gray palette matching figS1/fig2b style.
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
    ("gsd_airway_stenosis", "Airway stenosis"),
    ("gsd_laryngeal_dystonia", "Laryngeal dystonia"),
    ("gsd_depression", "Depression/MDD"),
    ("gsd_anxiety", "Anxiety disorder"),
    ("gsd_benign_lesion", "Benign vocal lesion"),
    ("gsd_mtd", "Muscle tension dysphonia"),
    ("gsd_copd_asthma", "COPD/Asthma"),
]


def collect(d, items):
    vw_m, vw_s, vh_m, vh_s = [], [], [], []
    for key, _ in items:
        vw = d.get(f"voicefm_whisper/{key}", [])
        vh = d.get(f"voicefm_hubert/{key}", [])
        vw_m.append(np.mean(vw) if vw else 0)
        vw_s.append(np.std(vw, ddof=1) if len(vw) > 1 else 0)
        vh_m.append(np.mean(vh) if vh else 0)
        vh_s.append(np.std(vh, ddof=1) if len(vh) > 1 else 0)
    return vw_m, vw_s, vh_m, vh_s


def auroc_color(val):
    if val >= 0.85:
        return "#E8F5E9"
    elif val >= 0.7:
        return "#FFF8E1"
    else:
        return "#FFEBEE"


def main():
    with open(RESULTS / "prospective_test_probes.json") as f:
        d = json.load(f)

    vw_m_c, vw_s_c, vh_m_c, vh_s_c = collect(d, CATEGORIES)
    vw_m_d, vw_s_d, vh_m_d, vh_s_d = collect(d, DIAGNOSES)

    fig = plt.figure(figsize=(16, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.45], wspace=0.18)

    # ── Panel a: Disease categories bar chart ──
    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(CATEGORIES))
    w = 0.38
    ax_a.bar(x - w / 2, vw_m_c, w, yerr=vw_s_c, color=BLUE, label="VoiceFM-Whisper",
             capsize=3, edgecolor="white", error_kw={"linewidth": 1.0})
    ax_a.bar(x + w / 2, vh_m_c, w, yerr=vh_s_c, color=GRAY, label="VoiceFM-HuBERT",
             capsize=3, edgecolor="white", error_kw={"linewidth": 1.0})
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([c[1] for c in CATEGORIES], fontsize=9, rotation=20, ha="right")
    ax_a.set_ylabel("AUROC (5-seed mean ± SD)", fontsize=10)
    ax_a.set_ylim(0.5, 1.08)
    ax_a.axhline(0.5, color="#E5E7EB", linewidth=1, linestyle="--", zorder=0)
    ax_a.legend(loc="upper right", fontsize=9, frameon=False)
    ax_a.grid(axis="y", alpha=0.15)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.text(-0.02, 1.02, "a", transform=ax_a.transAxes,
              fontsize=22, fontweight="bold", va="bottom", ha="right")
    ax_a.set_title("Disease categories", fontsize=11, pad=8)

    # ── Panel b: Individual diagnoses table ──
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_axis_off()

    header = ["Diagnosis", "VoiceFM-Whisper", "VoiceFM-HuBERT"]
    rows = []
    for (k, lbl), m_w, s_w, m_h, s_h in zip(DIAGNOSES, vw_m_d, vw_s_d, vh_m_d, vh_s_d):
        rows.append([
            lbl,
            f"{m_w:.3f} ± {s_w:.3f}",
            f"{m_h:.3f} ± {s_h:.3f}" if m_h > 0 else "—",
        ])

    table = ax_b.table(
        cellText=rows,
        colLabels=header,
        cellLoc="center",
        colWidths=[0.50, 0.25, 0.25],
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Style header
    for j in range(len(header)):
        cell = table[0, j]
        cell.set_facecolor("#2563EB")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        cell.set_edgecolor("white")

    # Style data rows: green/yellow/red fill on AUROC columns
    for i, (vw, vh) in enumerate(zip(vw_m_d, vh_m_d)):
        for j in range(len(header)):
            cell = table[i + 1, j]
            cell.set_edgecolor("#E0E0E0")
            if j == 1:
                cell.set_facecolor(auroc_color(vw))
            elif j == 2 and vh > 0:
                cell.set_facecolor(auroc_color(vh))
            else:
                cell.set_facecolor("white")
        table[i + 1, 0].set_text_props(ha="left")

    table[0, 0].set_text_props(ha="left")

    ax_b.text(-0.02, 1.02, "b", transform=ax_b.transAxes,
              fontsize=22, fontweight="bold", va="bottom", ha="right")
    ax_b.set_title("Individual diagnoses", fontsize=11, pad=8)

    plt.tight_layout()
    plt.savefig(PAPER_DIR / "fig3_prospective.png", dpi=300, bbox_inches="tight")
    plt.savefig(PAPER_DIR / "fig3_prospective.pdf", bbox_inches="tight")
    print("Saved to paper_v3/fig3_prospective.png")

    print("\nVoiceFM-Whisper (5-seed mean ± SD):")
    for (k, lbl), m, s in zip(DIAGNOSES, vw_m_d, vw_s_d):
        print(f"  {lbl}: {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    main()
