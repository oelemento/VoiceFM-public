#!/usr/bin/env python3
"""Figure 1b+c: Training curves for VoiceFM GSD model.

Parses SLURM training log to produce two-panel figure:
  Left (b): train/val loss vs epoch
  Right (c): Recall@5 vs epoch

Output: paper_v3/fig1bc_training.png

Usage:
    python scripts/paper_fig1b_training_v3.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent


def parse_training_log(log_path):
    """Parse epoch lines from training log."""
    epochs, train_loss, val_loss, r1, r5, temp = [], [], [], [], [], []

    pattern = re.compile(
        r"Epoch (\d+)/\d+.*?"
        r"train_loss: ([\d.]+),\s*"
        r"val_loss: ([\d.]+),\s*"
        r"R@1_a2c: ([\d.]+),\s*"
        r"R@5_a2c: ([\d.]+),\s*"
        r"temp: ([\d.]+)"
    )

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                val_loss.append(float(m.group(3)))
                r1.append(float(m.group(4)))
                r5.append(float(m.group(5)))
                temp.append(float(m.group(6)))

    return {
        "epoch": np.array(epochs),
        "train_loss": np.array(train_loss),
        "val_loss": np.array(val_loss),
        "r1": np.array(r1),
        "r5": np.array(r5),
        "temp": np.array(temp),
    }


def main():
    log_path = PROJECT_ROOT / "results_v3" / "training_log_whisper_ft4_v3_seed42.txt"
    if not log_path.exists():
        print(f"Training log not found: {log_path}")
        return

    data = parse_training_log(log_path)
    print(f"Parsed {len(data['epoch'])} epochs")
    print(f"  train_loss: {data['train_loss'][0]:.3f} → {data['train_loss'][-1]:.3f}")
    print(f"  val_loss:   {data['val_loss'][0]:.3f} → {data['val_loss'][-1]:.3f} (min={data['val_loss'].min():.3f})")
    print(f"  R@5:        {data['r5'][0]:.3f} → {data['r5'][-1]:.3f} (max={data['r5'].max():.3f})")

    # Find best epoch
    best_epoch = data["epoch"][np.argmin(data["val_loss"])]
    print(f"  Best epoch: {best_epoch}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Loss curves
    ax1.plot(data["epoch"], data["train_loss"], color="#2563EB", linewidth=1.5, label="Train loss", alpha=0.8)
    ax1.plot(data["epoch"], data["val_loss"], color="#DC2626", linewidth=1.5, label="Val loss", alpha=0.8)
    ax1.axvline(x=best_epoch, color="#9CA3AF", linestyle=":", linewidth=1, alpha=0.7)
    ax1.annotate(f"Best\n(epoch {best_epoch})", xy=(best_epoch, data["val_loss"].min()),
                 xytext=(best_epoch + 10, data["val_loss"].min() + 0.2),
                 fontsize=8, color="#6B7280",
                 arrowprops=dict(arrowstyle="->", color="#9CA3AF", lw=0.8))
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(fontsize=9, framealpha=0.95)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(alpha=0.2)
    ax1.text(-0.02, 1.02, "b", transform=ax1.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="right")

    # Right: R@5 curve
    ax2.plot(data["epoch"], data["r5"], color="#059669", linewidth=1.5, alpha=0.8)
    ax2.axvline(x=best_epoch, color="#9CA3AF", linestyle=":", linewidth=1, alpha=0.7)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Recall@5 (audio→clinical)", fontsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(alpha=0.2)
    ax2.text(-0.02, 1.02, "c", transform=ax2.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="right")

    plt.tight_layout()

    out_dir = PROJECT_ROOT / "paper_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "fig1bc_training.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "fig1bc_training.pdf", bbox_inches="tight")
    print(f"\nSaved to paper_v3/fig1bc_training.png")


if __name__ == "__main__":
    main()
