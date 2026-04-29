#!/usr/bin/env python3
"""Figure S4: t-SNE embedding visualization — VoiceFM-Whisper.

2×3 grid: VoiceFM-Whisper (top) vs Frozen Whisper (bottom),
colored by (a,d) diagnosis category, (b,e) severity, (c,f) voice quality.

Uses voicefm_whisper_embeddings.npz (seed 42) + frozen whisper embeddings.

Usage:
    python scripts/paper_figS4_tsne_v3.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results_v3"
PAPER_DIR = PROJECT_ROOT / "paper_v3"
PAPER_DIR.mkdir(exist_ok=True)

CAT_COLORS = {
    "control": "#4CAF50",
    "cat_voice": "#FF9800",
    "cat_neuro": "#F44336",
    "cat_mood": "#2196F3",
    "cat_respiratory": "#9C27B0",
}
CAT_LABELS = {
    "control": "Control",
    "cat_voice": "Voice",
    "cat_neuro": "Neurological",
    "cat_mood": "Mood",
    "cat_respiratory": "Respiratory",
}


def get_participant_category(row):
    for cat in ["cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]:
        if cat in row and int(row[cat]) == 1:
            return cat
    if "gsd_control" in row and int(row["gsd_control"]) == 1:
        return "control"
    return None


def main():
    # Load VoiceFM-Whisper embeddings
    cache = np.load(RESULTS / "voicefm_whisper_embeddings.npz", allow_pickle=True)
    pids = list(cache["pids"])
    vw_embs = cache["embeddings"]

    participants = pd.read_parquet(PROJECT_ROOT / "data" / "processed_v3" / "participants.parquet")

    # Get categories
    categories = []
    valid_idx = []
    for i, pid in enumerate(pids):
        if pid in participants.index:
            cat = get_participant_category(participants.loc[pid])
            if cat is not None:
                categories.append(cat)
                valid_idx.append(i)

    vw_valid = vw_embs[valid_idx]
    cats = np.array(categories)

    print(f"Valid participants: {len(valid_idx)}")
    for c in CAT_LABELS:
        print(f"  {CAT_LABELS[c]}: {np.sum(cats == c)}")

    # t-SNE on VoiceFM-Whisper
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, metric="cosine", random_state=42, max_iter=1000)
    vw_2d = tsne.fit_transform(vw_valid)

    # Plot: 1×3 (gender, diagnosis, severity)
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(18, 6))

    # Look up gender for each valid participant
    valid_pids = [pids[j] for j in valid_idx]
    gender_arr = np.array([int(participants.loc[p, "gender"]) if p in participants.index else -1
                            for p in valid_pids])
    GENDER_COLORS = {0: "#E91E63", 1: "#1E88E5", 2: "#9C27B0", 3: "#9E9E9E"}
    GENDER_LABELS = {0: "Female", 1: "Male", 2: "Non-binary", 3: "Other"}

    # Panel a: by gender (explains the 4 clusters)
    for code, lbl in GENDER_LABELS.items():
        mask = gender_arr == code
        n = int(np.sum(mask))
        if n == 0:
            continue
        ax_a.scatter(vw_2d[mask, 0], vw_2d[mask, 1], c=GENDER_COLORS[code],
                     label=f"{lbl} (n={n})",
                     s=15, alpha=0.7, edgecolors="none")
    ax_a.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    ax_a.set_title("VoiceFM-Whisper — by speaker gender", fontsize=11, fontweight="bold")
    ax_a.text(-0.02, 1.02, "a", transform=ax_a.transAxes,
              fontsize=14, fontweight="bold", va="bottom", ha="right")

    # Panel b: by diagnosis category
    for cat in ["control", "cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]:
        mask = cats == cat
        if np.sum(mask) == 0:
            continue
        ax_b.scatter(vw_2d[mask, 0], vw_2d[mask, 1], c=CAT_COLORS[cat],
                     label=f"{CAT_LABELS[cat]} (n={np.sum(mask)})",
                     s=15, alpha=0.7, edgecolors="none")
    ax_b.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax_b.set_xticks([]); ax_b.set_yticks([])
    ax_b.set_title("VoiceFM-Whisper — by diagnosis", fontsize=11, fontweight="bold")
    ax_b.text(-0.02, 1.02, "b", transform=ax_b.transAxes,
              fontsize=14, fontweight="bold", va="bottom", ha="right")

    # Panel c: by overall severity (CAPE-V, predominantly voice patients)
    capev_path = PROJECT_ROOT / "data" / "processed_v3" / "capev_scores.parquet"
    if capev_path.exists():
        capev = pd.read_parquet(capev_path)
        severity = []
        sev_idx = []
        for i, pid in enumerate(valid_pids):
            if pid in capev.index and "overall_severity" in capev.columns:
                s = capev.loc[pid, "overall_severity"]
                if np.isfinite(s):
                    severity.append(s); sev_idx.append(i)
        if len(sev_idx) > 20:
            sev_arr = np.array(severity)
            sc = ax_c.scatter(vw_2d[sev_idx, 0], vw_2d[sev_idx, 1], c=sev_arr,
                              cmap="RdYlGn_r", s=15, alpha=0.7, edgecolors="none",
                              vmin=0, vmax=100)
            plt.colorbar(sc, ax=ax_c, label="Overall severity", shrink=0.8)
            ax_c.set_title("VoiceFM-Whisper — by CAPE-V severity",
                           fontsize=11, fontweight="bold")
        else:
            ax_c.text(0.5, 0.5, "Insufficient severity data",
                      ha="center", transform=ax_c.transAxes)
    else:
        ax_c.text(0.5, 0.5, "No CAPE-V data", ha="center", transform=ax_c.transAxes)
    ax_c.set_xticks([]); ax_c.set_yticks([])
    ax_c.text(-0.02, 1.02, "c", transform=ax_c.transAxes,
              fontsize=14, fontweight="bold", va="bottom", ha="right")

    plt.tight_layout()
    plt.savefig(PAPER_DIR / "figS4_tsne.png", dpi=300, bbox_inches="tight")
    plt.savefig(PAPER_DIR / "figS4_tsne.pdf", bbox_inches="tight")
    print("Saved to paper/figS4_tsne.png")


if __name__ == "__main__":
    main()
