#!/usr/bin/env python3
"""Figure 2b: Per-diagnosis AUROC table — VoiceFM-Whisper vs VoiceFM-HuBERT.

Rendered as a matplotlib table image. Both models use 256d embeddings,
5-seed eval with create_participant_splits (70/15/15).

Usage:
    python scripts/paper_fig2b_diagnoses_v3.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results_v3"
PAPER_DIR = PROJECT_ROOT / "paper_v3"
PAPER_DIR.mkdir(exist_ok=True)

DIAGS = [
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


def main():
    pq = pd.read_parquet(PROJECT_ROOT / "data" / "processed_v3" / "participants.parquet")

    # All models from unified eval (identical methodology)
    unified_path = RESULTS / "unified_gsd_probes.json"
    with open(unified_path) as f:
        unified = json.load(f)

    # Extract per-diagnosis data for each model
    vw, vh, fw, fh = {}, {}, {}, {}
    for flag, _ in DIAGS:
        if f"voicefm_whisper/{flag}" in unified:
            vw[flag] = unified[f"voicefm_whisper/{flag}"]
        if f"voicefm_hubert/{flag}" in unified:
            vh[flag] = unified[f"voicefm_hubert/{flag}"]
        if f"frozen_whisper/{flag}" in unified:
            fw[flag] = unified[f"frozen_whisper/{flag}"]
        if f"frozen_hubert/{flag}" in unified:
            fh[flag] = unified[f"frozen_hubert/{flag}"]

    # Collect rows sorted by VoiceFM-Whisper AUROC desc
    diag_data = []
    for flag, label in DIAGS:
        n = int((pq[flag] == 1).sum()) if flag in pq.columns else 0
        if n < 5:
            continue
        vw_vals = vw.get(flag, [])
        vh_vals = vh.get(flag, [])
        fw_vals = fw.get(flag, [])
        fh_vals = fh.get(flag, [])
        if not vw_vals:
            continue
        vw_m, vw_s = np.mean(vw_vals), np.std(vw_vals)
        vh_m = np.mean(vh_vals) if vh_vals else float("nan")
        vh_s = np.std(vh_vals) if vh_vals else 0
        fw_m = np.mean(fw_vals) if fw_vals else float("nan")
        fw_s = np.std(fw_vals) if fw_vals else 0
        fh_m = np.mean(fh_vals) if fh_vals else float("nan")
        fh_s = np.std(fh_vals) if fh_vals else 0
        diag_data.append((flag, label, n, vw_m, vw_s, vh_m, vh_s, fw_m, fw_s, fh_m, fh_s))

    diag_data.sort(key=lambda x: -x[3])

    header = ["Diagnosis", "N", "VoiceFM-Whisper", "VoiceFM-HuBERT", "Frozen Whisper", "Frozen HuBERT"]
    rows = []
    for flag, label, n, vw_m, vw_s, vh_m, vh_s, fw_m, fw_s, fh_m, fh_s in diag_data:
        vw_str = f"{vw_m:.2f}±{vw_s:.2f}"
        vh_str = f"{vh_m:.2f}±{vh_s:.2f}" if not np.isnan(vh_m) else "—"
        fw_str = f"{fw_m:.2f}±{fw_s:.2f}" if not np.isnan(fw_m) else "—"
        fh_str = f"{fh_m:.2f}±{fh_s:.2f}" if not np.isnan(fh_m) else "—"
        rows.append([label, str(n), vw_str, vh_str, fw_str, fh_str])

    # Render as matplotlib table
    n_rows = len(rows)
    fig_h = 0.42 * (n_rows + 1) + 0.6
    fig, ax = plt.subplots(figsize=(12, fig_h), dpi=150)
    ax.set_axis_off()

    table = ax.table(
        cellText=rows,
        colLabels=header,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(header)):
        cell = table[0, j]
        cell.set_facecolor("#2563EB")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        cell.set_edgecolor("white")

    # Style data rows
    def auroc_color(val):
        if val >= 0.85:
            return "#E8F5E9"
        elif val >= 0.7:
            return "#FFF8E1"
        else:
            return "#FFEBEE"

    for i in range(n_rows):
        vw_m = diag_data[i][3]
        vh_m = diag_data[i][5]
        fw_m = diag_data[i][7]
        fh_m = diag_data[i][9]

        for j in range(len(header)):
            cell = table[i + 1, j]
            cell.set_edgecolor("#E0E0E0")
            if j == 2:
                cell.set_facecolor(auroc_color(vw_m))
            elif j == 3 and not np.isnan(vh_m):
                cell.set_facecolor(auroc_color(vh_m))
            elif j == 4 and not np.isnan(fw_m):
                cell.set_facecolor(auroc_color(fw_m))
            elif j == 5 and not np.isnan(fh_m):
                cell.set_facecolor(auroc_color(fh_m))
            else:
                cell.set_facecolor("white")

        table[i + 1, 0].set_text_props(ha="left")

    table[0, 0].set_text_props(ha="left")

    col_widths = [0.22, 0.05, 0.16, 0.16, 0.16, 0.16]
    for j, w in enumerate(col_widths):
        for i in range(n_rows + 1):
            table[i, j].set_width(w)

    ax.text(0.02, 0.98, "b", fontsize=20, fontweight="bold", transform=ax.transAxes,
            va="top", ha="left")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(PAPER_DIR / f"fig2b_diagnoses.{ext}", bbox_inches="tight",
                    facecolor="white", pad_inches=0.1, dpi=300)
    print("Saved to paper/fig2b_diagnoses.png")
    plt.close()

    print(f"\n{'Diagnosis':<28} {'N':>4} {'VW':>12} {'VH':>12} {'FW':>12} {'FH':>12}")
    print("-" * 86)
    for flag, label, n, vw_m, vw_s, vh_m, vh_s, fw_m, fw_s, fh_m, fh_s in diag_data:
        fw_str = f"{fw_m:.3f}±{fw_s:.3f}" if not np.isnan(fw_m) else "—"
        fh_str = f"{fh_m:.3f}±{fh_s:.3f}" if not np.isnan(fh_m) else "—"
        print(f"{label:<28} {n:>4} {vw_m:.3f}±{vw_s:.3f} {vh_m:.3f}±{vh_s:.3f} {fw_str:>12} {fh_str:>12}")


if __name__ == "__main__":
    main()
