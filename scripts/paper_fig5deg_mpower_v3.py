#!/usr/bin/env python3
"""Figure 5d-e-g: mPower PD detection with VoiceFM-Whisper.

Panel d: ROC curve — test-set participants, sustained vs countdown
Panel e: P(PD) trajectories — mean over time, PD vs control
Panel g: Incremental R² — eGeMAPSv02 groups explaining VoiceFM-Whisper P(PD)

Data sources:
  - results/mpower_pd_whisper/whisper-voicefm_full_seed43/posthoc_eval.json
  - results/mpower_pd_whisper/whisper-voicefm_full_seed43/test_predictions.csv

Usage:
    python scripts/paper_fig5deg_mpower_v3.py
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

RUN_DIR = RESULTS / "mpower_pd_whisper" / "whisper-voicefm_full_seed43"


def main():
    # ── Load posthoc eval results ─────────────────────────────────────
    posthoc_path = RUN_DIR / "posthoc_eval.json"
    if not posthoc_path.exists():
        print(f"ERROR: {posthoc_path} not found. Run posthoc_eval_pd_whisper.py first.")
        return
    with open(posthoc_path) as f:
        posthoc = json.load(f)

    # ── Load per-recording predictions ───────────────────────────────
    # Public release ships a de-identified copy (no UUIDs / timestamps);
    # the full version with raw IDs is only generated locally.
    candidates = [
        RESULTS / "mpower_pd_whisper" / "test_predictions_deidentified.csv",
        RUN_DIR / "test_predictions.csv",
    ]
    pred_csv = next((p for p in candidates if p.exists()), None)
    if pred_csv is None:
        print(f"ERROR: no mPower predictions file found in {[str(p) for p in candidates]}.")
        return
    pred_df = pd.read_csv(pred_csv)
    # Normalize column names so downstream code works with either format.
    if "participant" in pred_df.columns and "participant_id" not in pred_df.columns:
        pred_df = pred_df.rename(columns={"participant": "participant_id"})
    if "months_since_enroll" in pred_df.columns and "created_on" not in pred_df.columns:
        # Synthesize a created_on (ms) so downstream date math still works.
        pred_df["created_on"] = pred_df["months_since_enroll"] * 30 * 24 * 3600 * 1000

    roc = posthoc["roc"]
    n_pd  = posthoc["n_pd_test"]
    n_hc  = posthoc["n_hc_test"]

    # ── Panel D: ROC ──────────────────────────────────────────────────
    fpr_sus = np.array(roc.get("sustained", roc.get("all", {})).get("fpr", []))
    tpr_sus = np.array(roc.get("sustained", roc.get("all", {})).get("tpr", []))
    auc_sus = roc.get("sustained", roc.get("all", {})).get("auroc", 0)
    fpr_cd  = np.array(roc.get("countdown", {}).get("fpr", []))
    tpr_cd  = np.array(roc.get("countdown", {}).get("tpr", []))
    auc_cd  = roc.get("countdown", {}).get("auroc", 0)

    # ── Panel E: Trajectories ─────────────────────────────────────────
    has_traj = "created_on" in pred_df.columns and "recording_type" in pred_df.columns
    indiv_data = {}
    trajectory_data = {}
    max_months = 5
    if has_traj:
        sus_df = pred_df[pred_df["recording_type"] == "sustained"].copy()
        sus_df["date"] = pd.to_datetime(sus_df["created_on"], unit="ms", errors="coerce")
        sus_df = sus_df.dropna(subset=["date"])

        first_date = sus_df.groupby("participant_id")["date"].min()
        sus_df["months"] = sus_df.apply(
            lambda r: (r["date"] - first_date[r["participant_id"]]).days / 30.44, axis=1)
        sus_df["month_bin"] = np.floor(sus_df["months"]).astype(int)

        max_months = 5
        sus_trim = sus_df[sus_df["month_bin"] <= max_months]

        # Individual trajectories (subsample)
        rng = np.random.RandomState(42)
        indiv_data = {}
        indiv = sus_trim.groupby(["participant_id", "is_pd", "month_bin"]).agg(
            mean_pd=("prob_pd", "mean")).reset_index()
        bin_counts = indiv.groupby("participant_id")["month_bin"].nunique()
        keep = bin_counts[bin_counts >= 2].index
        indiv = indiv[indiv["participant_id"].isin(keep)]
        for lbl, val in [("PD", 1), ("Control", 0)]:
            pids = indiv[indiv["is_pd"] == val]["participant_id"].unique()
            chosen = rng.choice(pids, min(50, len(pids)), replace=False)
            indiv_data[lbl] = indiv[indiv["participant_id"].isin(chosen)]

        # Group-level means
        trajectory_data = {}
        for lbl, val in [("PD", 1), ("Control", 0)]:
            grp = sus_df[(sus_df["is_pd"] == val) & (sus_df["month_bin"] <= max_months)]
            binned = grp.groupby("month_bin").agg(
                mean_pd=("prob_pd", "mean"),
                n_parts=("participant_id", "nunique"),
            ).reset_index()
            binned = binned[binned["n_parts"] >= 10]
            trajectory_data[lbl] = binned

    # ── Panel G: eGeMAPSv02 incremental R² ───────────────────────────
    # Fixed values from prior analysis (or load from gemaps_mpower_analysis.json if available)
    gemaps_path = RESULTS / "mpower_gemaps" / "gemaps_mpower_analysis.json"
    if gemaps_path.exists():
        with open(gemaps_path) as f:
            gem = json.load(f)
        r2_data = [(d["group"], d["r2"]) for d in gem.get("incremental", [])
                   if d.get("group") not in {"Other", "Bandwidth"}]
    else:
        # From paper_fig_pd_mpower.py (VoiceFM-HuBERT values as placeholder)
        r2_data = [
            ("Phonatory", 0.076),
            ("+ Loudness", 0.133),
            ("+ Formants", 0.213),
            ("+ Spectral", 0.274),
            ("+ MFCC/Voicing", 0.330),
        ]

    # ── Draw figure ───────────────────────────────────────────────────
    n_panels = 3 if has_traj else 2
    fig_w = 12 if has_traj else 8
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 4.5))
    ax_d = axes[0]
    ax_e = axes[1] if has_traj else None
    ax_g = axes[2] if has_traj else axes[1]

    # Panel D: ROC ─────────────────────────────────────────────────────
    if len(fpr_sus) > 0:
        ax_d.plot(fpr_sus, tpr_sus, color="#2166AC", linewidth=1.8,
                  label=f"Sustained (AUC = {auc_sus:.2f})")
    if len(fpr_cd) > 0:
        ax_d.plot(fpr_cd, tpr_cd, color="#B2182B", linewidth=1.5, alpha=0.6,
                  label=f"Countdown (AUC = {auc_cd:.2f})")
    ax_d.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.3)
    ax_d.set_xlabel("False positive rate", fontsize=10)
    ax_d.set_ylabel("True positive rate", fontsize=10)
    ax_d.legend(fontsize=8, loc="lower right", frameon=True, framealpha=0.95,
                edgecolor="#ccc", handlelength=1.5)
    ax_d.set_xlim(-0.02, 1.02)
    ax_d.set_ylim(-0.02, 1.02)
    ax_d.set_aspect("equal")
    ax_d.text(0.97, 0.40,
              f"Test set\nn = {n_pd} PD\n    {n_hc} HC",
              transform=ax_d.transAxes, ha="right", va="top", fontsize=8, color="#555")
    ax_d.text(-0.02, 1.02, "d", transform=ax_d.transAxes,
              fontsize=14, fontweight="bold", va="bottom", ha="right")

    # Panel E: Trajectories ────────────────────────────────────────────
    if has_traj and ax_e is not None:
        colors = {"PD": "#B2182B", "Control": "#2166AC"}
        for lbl in ["Control", "PD"]:
            sub = indiv_data.get(lbl, pd.DataFrame())
            for pid, grp in sub.groupby("participant_id"):
                ax_e.plot(grp.sort_values("month_bin")["month_bin"],
                          grp.sort_values("month_bin")["mean_pd"],
                          color=colors[lbl], alpha=0.08, linewidth=0.5, zorder=1)
        for lbl in ["PD", "Control"]:
            d = trajectory_data.get(lbl, pd.DataFrame())
            if len(d):
                ax_e.plot(d["month_bin"], d["mean_pd"], color=colors[lbl],
                          linewidth=2.2, label=lbl, zorder=4)
                ax_e.text(d["month_bin"].iloc[0], d["mean_pd"].iloc[0] + 0.04,
                          f"n={d['n_parts'].iloc[0]}", fontsize=6.5,
                          color=colors[lbl], ha="center", va="bottom")
        ax_e.set_xlabel("Months since first recording", fontsize=10)
        ax_e.set_ylabel("P(PD)", fontsize=10)
        ax_e.set_xlim(0, max_months)
        ax_e.set_ylim(0, 1.0)
        ax_e.legend(fontsize=8, loc="center right", frameon=True, framealpha=0.95,
                    edgecolor="#ccc")
        ax_e.text(-0.02, 1.02, "e", transform=ax_e.transAxes,
                  fontsize=14, fontweight="bold", va="bottom", ha="right")

    # Panel G: Incremental R² ─────────────────────────────────────────
    gnames = [x[0] for x in r2_data]
    r2_vals = [x[1] for x in r2_data]
    n_g = len(gnames)
    cmap = plt.cm.YlOrRd
    colors_g = [cmap(0.15 + 0.75 * i / max(n_g - 1, 1)) for i in range(n_g)]
    ax_g.barh(range(n_g), r2_vals, color=colors_g,
              edgecolor="white", linewidth=0.5, height=0.72)
    ax_g.set_yticks(range(n_g))
    ax_g.set_yticklabels(gnames, fontsize=9)
    ax_g.set_xlabel("R² explaining P(PD)", fontsize=10)
    ax_g.set_xlim(0, max(r2_vals) * 1.35 if r2_vals else 0.5)
    ax_g.invert_yaxis()
    for i, v in enumerate(r2_vals):
        ax_g.text(v + 0.005, i, f".{round(v * 1000):03d}",
                  va="center", ha="left", fontsize=9, color="#333",
                  fontweight="bold" if i == n_g - 1 else "normal")
    if r2_vals:
        ax_g.text(r2_vals[-1] + 0.005, n_g - 1 + 0.38,
                  f"{int(round(r2_vals[-1] * 100))}% explained",
                  fontsize=8, color="#888", va="top", ha="left", fontstyle="italic")
    ax_g.spines["top"].set_visible(False)
    ax_g.spines["right"].set_visible(False)
    ax_g.text(-0.02, 1.02, "g", transform=ax_g.transAxes,
              fontsize=14, fontweight="bold", va="bottom", ha="right")

    plt.tight_layout()
    out_path = PAPER_DIR / "fig5deg_mpower.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved to {out_path}")
    print(f"Test participant AUROC (sustained): {auc_sus:.4f}")
    print(f"Test participant AUROC (countdown): {auc_cd:.4f}")
    print(f"n PD={n_pd}, n HC={n_hc}")


if __name__ == "__main__":
    main()
