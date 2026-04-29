#!/usr/bin/env python3
"""Evaluate Whisper multi-task model on GSD classification.

Loads a trained WhisperClinicalModel checkpoint, extracts 1280d embeddings,
runs logistic regression probes, and compares with frozen Whisper baseline.

Usage:
    python scripts/evaluate_whisper_multitask.py --checkpoint checkpoints_exp_whisper_mt_gsd_seed42/best_model.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from src.data.sampler import create_participant_splits
from src.data.audio_dataset import build_task_type_map
from src.data.clinical_encoder import ClinicalFeatureProcessor
from src.models.whisper_multitask_model import WhisperClinicalModel
from src.utils.preprocessing import load_and_preprocess, MAX_SAMPLES

LABELS = ["is_control_participant", "cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]
LABEL_TO_INPUT = {
    "is_control_participant": "gsd_control",
    "cat_voice": "cat_voice",
    "cat_neuro": "cat_neuro",
    "cat_mood": "cat_mood",
    "cat_respiratory": "cat_respiratory",
}
BATCH_SIZE = 8

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_embeddings(model, recordings, participants, device):
    """Extract 1280d embeddings per participant from trained model."""
    model.eval()
    audio_dir = PROJECT / "data" / "audio"
    task_type_map = build_task_type_map(recordings)
    pid_embeddings = {}

    items = []
    for _, row in recordings.iterrows():
        audio_path = audio_dir / row["audio_filename"]
        if audio_path.exists():
            items.append((row["record_id"], audio_path, row.get("recording_name", "")))

    logger.info("  %d recordings to process", len(items))

    for batch_start in range(0, len(items), BATCH_SIZE):
        if batch_start % 2000 == 0 and batch_start > 0:
            logger.info("  Processing %d/%d...", batch_start, len(items))

        batch_items = items[batch_start:batch_start + BATCH_SIZE]
        batch_wavs, batch_pids, batch_tasks = [], [], []

        for pid, audio_path, task_name in batch_items:
            try:
                wav = load_and_preprocess(str(audio_path), max_samples=MAX_SAMPLES)
            except Exception:
                continue
            if isinstance(wav, torch.Tensor):
                wav = wav.numpy()
            if len(wav) < 400:
                continue
            batch_wavs.append(wav)
            batch_pids.append(pid)
            batch_tasks.append(task_type_map.get(task_name, 0))

        if not batch_wavs:
            continue

        # Pad to max length in batch
        max_len = max(len(w) for w in batch_wavs)
        padded = np.zeros((len(batch_wavs), max_len), dtype=np.float32)
        masks = np.zeros((len(batch_wavs), max_len), dtype=np.int64)
        for i, wav in enumerate(batch_wavs):
            padded[i, :len(wav)] = wav
            masks[i, :len(wav)] = 1

        audio_tensor = torch.tensor(padded, device=device)
        mask_tensor = torch.tensor(masks, device=device)
        task_tensor = torch.tensor(batch_tasks, dtype=torch.long, device=device)

        with torch.no_grad():
            output = model(audio_tensor, mask_tensor, task_tensor)
            pooled = output["pooled"]  # (B, 1280)

        for i, pid in enumerate(batch_pids):
            if pid not in pid_embeddings:
                pid_embeddings[pid] = []
            pid_embeddings[pid].append(pooled[i].cpu().numpy())

    # Mean-pool per participant
    result = {}
    for pid, embs in pid_embeddings.items():
        result[pid] = np.mean(embs, axis=0)
    return result


def run_probe(X_train, y_train, X_test, y_test):
    if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
        return np.nan
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_tr, y_train)
    prob = clf.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_test, prob)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--experiment", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    config = {}
    model_path = PROJECT / "configs" / "whisper_multitask_model.yaml"
    with open(model_path) as f:
        config["model"] = yaml.safe_load(f)
    for name in ["data", "train"]:
        path = PROJECT / "configs" / f"{name}.yaml"
        if path.exists():
            with open(path) as f:
                config[name] = yaml.safe_load(f)

    if args.experiment:
        exp_path = PROJECT / "configs" / "experiments" / f"{args.experiment}.yaml"
        if exp_path.exists():
            with open(exp_path) as f:
                overrides = yaml.safe_load(f) or {}
            for section in ["model", "data", "train"]:
                if section in overrides:
                    from scripts.train_whisper_multitask import deep_merge
                    config[section] = deep_merge(config[section], overrides[section])

    # Load data
    participants = pd.read_parquet(PROJECT / "data" / "processed" / "participants.parquet")
    recordings = pd.read_parquet(PROJECT / "data" / "processed" / "recordings.parquet")

    task_type_map = build_task_type_map(recordings)
    config["model"]["task_conditioning"]["num_task_types"] = len(task_type_map) + 1

    # Load model
    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = WhisperClinicalModel(config["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    logger.info("Model loaded")

    # Get splits
    split_cfg = config["data"]["splits"]
    train_ids, val_ids, test_ids = create_participant_splits(
        participants,
        train_ratio=split_cfg["train"], val_ratio=split_cfg["val"], test_ratio=split_cfg["test"],
        seed=split_cfg["seed"], stratify_col=split_cfg.get("stratify_by"),
    )

    train_rec = recordings[recordings["record_id"].isin(train_ids)]
    test_rec = recordings[recordings["record_id"].isin(test_ids)]

    # Extract embeddings
    logger.info("Extracting train embeddings (%d participants)...", len(train_ids))
    train_embs = extract_embeddings(model, train_rec, participants, device)
    logger.info("Extracting test embeddings (%d participants)...", len(test_ids))
    test_embs = extract_embeddings(model, test_rec, participants, device)

    # Probes
    train_avail = [p for p in train_ids if p in train_embs]
    test_avail = [p for p in test_ids if p in test_embs]
    X_train = np.array([train_embs[p] for p in train_avail])
    X_test = np.array([test_embs[p] for p in test_avail])
    train_df = participants.loc[train_avail]
    test_df = participants.loc[test_avail]

    logger.info("Probes: train=%d, test=%d, dim=%d", len(train_avail), len(test_avail), X_train.shape[1])

    results = {}
    for label in LABELS:
        y_train = train_df[label].values.astype(int)
        y_test = test_df[label].values.astype(int)
        auroc = run_probe(X_train, y_train, X_test, y_test)
        results[f"whisper_mt/probe/{label}/auroc"] = auroc
        logger.info("  %s: AUROC=%.3f", label, auroc)

    mean_auroc = np.mean([results[f"whisper_mt/probe/{l}/auroc"] for l in LABELS])
    logger.info("Mean AUROC: %.3f", mean_auroc)

    # Save
    out_dir = Path(args.checkpoint).parent
    out_path = out_dir / "eval_results_best_model.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
