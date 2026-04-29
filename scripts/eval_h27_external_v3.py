#!/usr/bin/env python3
"""Evaluate H27 (frozen Whisper contrastive 256d) on external datasets.

Runs all 5 seeds on the specified dataset, reports mean ± SD.
Also evaluates frozen Whisper 1280d baseline for comparison.

Usage:
    python -u scripts/eval_h27_external_v3.py --dataset neurovoz
    python -u scripts/eval_h27_external_v3.py --dataset coswara
    python -u scripts/eval_h27_external_v3.py --dataset svd
    python -u scripts/eval_h27_external_v3.py --dataset mdvr_kcl
    python -u scripts/eval_h27_external_v3.py --dataset mpower
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from src.models import build_audio_encoder
from src.utils.preprocessing import load_and_preprocess, MAX_SAMPLES

SEEDS = list(range(42, 47))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_whisper_frozen(device):
    """Load frozen Whisper encoder for 1280d baseline."""
    from transformers import WhisperModel, WhisperFeatureExtractor
    model = WhisperModel.from_pretrained("openai/whisper-large-v2", torch_dtype=torch.float32)
    encoder = model.encoder.float().to(device).eval()
    del model.decoder
    for p in encoder.parameters():
        p.requires_grad = False
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")
    return encoder, fe


def load_h27_encoder(seed, device):
    """Load H27 VoiceFM audio encoder (frozen Whisper + contrastive projection)."""
    ckpt_path = PROJECT / f"checkpoints_exp_whisper_gsd_v3_seed{seed}" / "best_model.pt"
    if not ckpt_path.exists():
        return None

    with open(PROJECT / "configs" / "model.yaml") as f:
        model_cfg = yaml.safe_load(f)

    # Override to whisper
    model_cfg["audio_encoder"]["type"] = "whisper"
    model_cfg["audio_encoder"]["backbone"] = "openai/whisper-large-v2"
    model_cfg["audio_encoder"]["freeze_backbone"] = True
    model_cfg["audio_encoder"]["unfreeze_last_n"] = 0

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


def load_dataset(dataset_name):
    """Load external dataset metadata and audio directory."""
    ext_dir = PROJECT / "data" / "external"

    if dataset_name == "neurovoz":
        meta = pd.read_csv(ext_dir / "neurovoz" / "metadata.csv")
        audio_dir = ext_dir / "neurovoz" / "data" / "audios"
        return meta, audio_dir, "participant_id", "label"

    elif dataset_name == "coswara":
        meta = pd.read_csv(ext_dir / "coswara" / "metadata.csv")
        audio_dir = ext_dir / "coswara" / "audio"
        return meta, audio_dir, "participant_id", "is_covid"

    elif dataset_name == "svd":
        meta = pd.read_csv(ext_dir / "svd" / "metadata.csv")
        audio_dir = ext_dir / "svd" / "audio"
        pid_col = "speaker_id" if "speaker_id" in meta.columns else "participant_id"
        return meta, audio_dir, pid_col, "is_pathological"

    elif dataset_name == "mdvr_kcl":
        meta = pd.read_csv(ext_dir / "mdvr_kcl" / "metadata.csv")
        audio_dir = ext_dir / "mdvr_kcl" / "audio"
        return meta, audio_dir, "subject_id", "label"

    elif dataset_name == "mpower":
        meta = pd.read_csv(PROJECT / "data" / "mpower" / "mpower_metadata.csv")
        audio_dir = PROJECT / "data" / "mpower" / "audio"
        return meta, audio_dir, "participant_id", "is_pd"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def extract_embeddings_h27(encoder, meta, audio_dir, pid_col, device):
    """Extract H27 256d embeddings per participant."""
    pid_embs = {}
    fname_col = "filename" if "filename" in meta.columns else "audio_filename"

    for idx, (_, row) in enumerate(meta.iterrows()):
        if idx % 200 == 0 and idx > 0:
            logger.info("    %d/%d...", idx, len(meta))

        pid = str(row[pid_col])
        audio_path = audio_dir / row[fname_col]
        if not audio_path.exists():
            continue

        wav = load_and_preprocess(audio_path)  # handles m4a, resamples to 16kHz
        if wav.numel() < 400:
            continue

        audio_t = wav.unsqueeze(0).to(device)
        mask_t = torch.ones(1, wav.shape[0], dtype=torch.long, device=device)
        task_t = torch.zeros(1, dtype=torch.long, device=device)

        with torch.no_grad():
            emb = encoder(audio_input_values=audio_t, attention_mask=mask_t, task_type_ids=task_t)

        pid_embs.setdefault(pid, []).append(emb[0].cpu().numpy())

    return {pid: np.mean(embs, axis=0) for pid, embs in pid_embs.items()}


def extract_embeddings_whisper(encoder, fe, meta, audio_dir, pid_col, device):
    """Extract frozen Whisper 1280d embeddings per participant."""
    pid_embs = {}
    fname_col = "filename" if "filename" in meta.columns else "audio_filename"

    for idx, (_, row) in enumerate(meta.iterrows()):
        if idx % 200 == 0 and idx > 0:
            logger.info("    %d/%d...", idx, len(meta))

        pid = str(row[pid_col])
        audio_path = audio_dir / row[fname_col]
        if not audio_path.exists():
            continue

        wav = load_and_preprocess(audio_path)  # handles m4a, resamples to 16kHz
        if wav.numel() < 400:
            continue

        wav_np = wav.numpy()
        mel = fe(wav_np, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            out = encoder(mel.float(), return_dict=True)
            token_len = max(1, int(len(wav_np) / MAX_SAMPLES * out.last_hidden_state.shape[1]))
            token_len = min(token_len, out.last_hidden_state.shape[1])
            pooled = out.last_hidden_state[0, :token_len, :].mean(dim=0).cpu().numpy()

        pid_embs.setdefault(pid, []).append(pooled)

    return {pid: np.mean(embs, axis=0) for pid, embs in pid_embs.items()}


def eval_5fold(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aurocs = []
    for tr_idx, te_idx in skf.split(X, y):
        if len(np.unique(y[te_idx])) < 2:
            continue
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(scaler.fit_transform(X[tr_idx]), y[tr_idx])
        prob = clf.predict_proba(scaler.transform(X[te_idx]))[:, 1]
        aurocs.append(float(roc_auc_score(y[te_idx], prob)))
    return float(np.mean(aurocs)), float(np.std(aurocs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["neurovoz", "coswara", "svd", "mdvr_kcl", "mpower"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta, audio_dir, pid_col, label_col = load_dataset(args.dataset)
    logger.info("Dataset: %s, %d recordings", args.dataset, len(meta))

    results = {}

    # Frozen Whisper 1280d baseline (extract once)
    logger.info("\n=== Frozen Whisper 1280d ===")
    w_encoder, w_fe = load_whisper_frozen(device)
    w_embs = extract_embeddings_whisper(w_encoder, w_fe, meta, audio_dir, pid_col, device)
    del w_encoder
    torch.cuda.empty_cache()

    pids = sorted(w_embs.keys())
    pid_labels = {}
    for _, row in meta.iterrows():
        pid = str(row[pid_col])
        pid_labels[pid] = int(row[label_col])

    pids_with_labels = [p for p in pids if p in pid_labels]
    X_w = np.array([w_embs[p] for p in pids_with_labels])
    y = np.array([pid_labels[p] for p in pids_with_labels])

    auroc, std = eval_5fold(X_w, y)
    results["frozen_whisper_1280d"] = {"auroc": auroc, "std": std, "n": len(pids_with_labels)}
    logger.info("  Frozen Whisper: AUROC=%.3f ± %.3f (n=%d)", auroc, std, len(pids_with_labels))

    # H27 per seed
    for seed in SEEDS:
        logger.info("\n=== H27 seed %d ===", seed)
        h27_encoder = load_h27_encoder(seed, device)
        if h27_encoder is None:
            logger.warning("  Checkpoint not found for seed %d", seed)
            continue

        h27_embs = extract_embeddings_h27(h27_encoder, meta, audio_dir, pid_col, device)
        del h27_encoder
        torch.cuda.empty_cache()

        pids_h27 = [p for p in pids_with_labels if p in h27_embs]
        X_h27 = np.array([h27_embs[p] for p in pids_h27])
        y_h27 = np.array([pid_labels[p] for p in pids_h27])

        auroc, std = eval_5fold(X_h27, y_h27)
        results.setdefault("h27_whisper_contrastive_256d", []).append({"auroc": auroc, "std": std, "seed": seed})
        logger.info("  H27 seed%d: AUROC=%.3f ± %.3f (n=%d)", seed, auroc, std, len(pids_h27))

    # Summary
    h27_aurocs = [r["auroc"] for r in results.get("h27_whisper_contrastive_256d", [])]
    logger.info("\n=== SUMMARY: %s ===", args.dataset)
    logger.info("  Frozen Whisper 1280d: %.3f", results["frozen_whisper_1280d"]["auroc"])
    if h27_aurocs:
        logger.info("  H27 contrastive 256d: %.3f ± %.3f (%d seeds)",
                    np.mean(h27_aurocs), np.std(h27_aurocs), len(h27_aurocs))

    out_path = PROJECT / "results_v3" / f"eval_h27_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
