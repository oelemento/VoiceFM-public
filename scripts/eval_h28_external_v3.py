#!/usr/bin/env python3
"""Evaluate H28 (Whisper FT4 contrastive 256d) on external datasets.

Same as eval_h27_external.py but uses checkpoints_exp_whisper_ft4_gsd_v3_seed{seed}.

Usage:
    python -u scripts/eval_h28_external_v3.py --dataset neurovoz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

# Reuse everything from H27 script
from scripts.eval_h27_external import (
    load_whisper_frozen, load_dataset, extract_embeddings_h27,
    extract_embeddings_whisper, eval_5fold, logger, SEEDS,
)
from src.models import build_audio_encoder


def load_h28_encoder(seed, device):
    """Load H28 VoiceFM audio encoder (Whisper FT4 + contrastive projection)."""
    ckpt_path = PROJECT / f"checkpoints_exp_whisper_ft4_gsd_v3_seed{seed}" / "best_model.pt"
    if not ckpt_path.exists():
        return None

    with open(PROJECT / "configs" / "model.yaml") as f:
        model_cfg = yaml.safe_load(f)

    model_cfg["audio_encoder"]["type"] = "whisper"
    model_cfg["audio_encoder"]["backbone"] = "openai/whisper-large-v2"
    model_cfg["audio_encoder"]["freeze_backbone"] = True
    model_cfg["audio_encoder"]["unfreeze_last_n"] = 4  # FT4

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    num_task_types = state.get("audio_encoder.task_embedding.weight", torch.zeros(100, 1)).shape[0]

    audio_encoder = build_audio_encoder(
        config=model_cfg["audio_encoder"], num_task_types=num_task_types,
    )
    ae_state = {k.replace("audio_encoder.", "", 1): v for k, v in state.items()
                if k.startswith("audio_encoder.")}
    audio_encoder.load_state_dict(ae_state)
    return audio_encoder.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["neurovoz", "coswara", "svd", "mdvr_kcl", "mpower"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Override SEEDS list (e.g. --seeds 44 45 46 to skip 42 43).")
    args = parser.parse_args()
    seeds = args.seeds if args.seeds else SEEDS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta, audio_dir, pid_col, label_col = load_dataset(args.dataset)
    logger.info("Dataset: %s, %d recordings", args.dataset, len(meta))

    fname_col = "filename" if "filename" in meta.columns else "audio_filename"
    results = {}

    # Build pid->label mapping
    pid_labels = {}
    for _, row in meta.iterrows():
        pid_labels[str(row[pid_col])] = int(row[label_col])

    # H28 per seed
    for seed in seeds:
        logger.info("\n=== H28 seed %d ===", seed)
        encoder = load_h28_encoder(seed, device)
        if encoder is None:
            logger.warning("  Checkpoint not found for seed %d", seed)
            continue

        embs = extract_embeddings_h27(encoder, meta, audio_dir, pid_col, device)
        del encoder
        torch.cuda.empty_cache()

        pids = [p for p in sorted(embs.keys()) if p in pid_labels]
        X = np.array([embs[p] for p in pids])
        y = np.array([pid_labels[p] for p in pids])

        auroc, std = eval_5fold(X, y)
        results.setdefault("h28_whisper_ft4_256d", []).append({"auroc": auroc, "std": std, "seed": seed})
        logger.info("  H28 seed%d: AUROC=%.3f ± %.3f (n=%d)", seed, auroc, std, len(pids))

    # Summary
    h28_aurocs = [r["auroc"] for r in results.get("h28_whisper_ft4_256d", [])]
    logger.info("\n=== SUMMARY: %s ===", args.dataset)
    if h28_aurocs:
        logger.info("  H28 FT4 contrastive 256d: %.3f ± %.3f (%d seeds)",
                    np.mean(h28_aurocs), np.std(h28_aurocs), len(h28_aurocs))

    out_path = PROJECT / "results_v3" / f"eval_h28_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
