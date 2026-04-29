#!/usr/bin/env python3
"""Evaluate frozen Whisper on TAUKADIAL (MCI vs NC detection).

Extracts 1280d Whisper embeddings from train+test, runs 5-fold CV
on combined data, and also train-on-train/eval-on-test.

Usage:
    python -u scripts/evaluate_taukadial_v3.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

TAUKADIAL_DIR = PROJECT / "data" / "external" / "taukadial" / "TAUKADIAL-24"
BATCH_SIZE = 4  # small due to long recordings chunked to 30s

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_whisper_encoder(device):
    from transformers import WhisperModel, WhisperFeatureExtractor
    logger.info("Loading Whisper large-v2...")
    model = WhisperModel.from_pretrained("openai/whisper-large-v2")
    encoder = model.encoder.to(device).eval()
    del model.decoder
    for p in encoder.parameters():
        p.requires_grad = False
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")
    logger.info("Loaded (%dM params)", sum(p.numel() for p in encoder.parameters()) // 1_000_000)
    return encoder, fe


def extract_embeddings(encoder, fe, audio_dir, metadata, device):
    """Extract 1280d Whisper embeddings, chunking long recordings into 30s segments."""
    pid_embs = {}
    total = len(metadata)

    for idx, (_, row) in enumerate(metadata.iterrows()):
        if idx % 30 == 0:
            logger.info("  Processing %d/%d...", idx, total)

        fname = row["tkdname"] if "tkdname" in row else row.iloc[0]
        pid = fname.replace(".wav", "").rsplit("-", 1)[0]

        wav_path = audio_dir / fname
        if not wav_path.exists():
            logger.warning("Missing audio: %s", fname)
            continue

        wav, sr = sf.read(wav_path)
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)  # stereo to mono
        if sr != 16000:
            gcd = math.gcd(sr, 16000)
            wav = resample_poly(wav, 16000 // gcd, sr // gcd).astype(np.float32)

        # Chunk into 30s segments
        chunk_samples = 16000 * 30
        chunks = []
        for start in range(0, len(wav), chunk_samples):
            chunk = wav[start:start + chunk_samples]
            if len(chunk) < 8000:  # skip < 0.5s
                continue
            chunks.append(chunk.astype(np.float32))

        if not chunks:
            continue

        # Process chunks
        chunk_embs = []
        for chunk in chunks:
            mel = fe(chunk, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            with torch.no_grad():
                out = encoder(mel.float(), return_dict=True)
                hidden = out.last_hidden_state  # (1, 1500, 1280)
                # Masked mean-pool
                token_len = max(1, int(len(chunk) / chunk_samples * hidden.shape[1]))
                token_len = min(token_len, hidden.shape[1])
                pooled = hidden[0, :token_len, :].mean(dim=0)
                chunk_embs.append(pooled.cpu().numpy())

        # Mean across chunks for this recording
        rec_emb = np.mean(chunk_embs, axis=0)

        if pid not in pid_embs:
            pid_embs[pid] = []
        pid_embs[pid].append(rec_emb)

    # Mean-pool per participant (across 3 recordings)
    result = {}
    for pid, embs in pid_embs.items():
        result[pid] = np.mean(embs, axis=0)

    return result


def run_cv(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aurocs, accs = [], []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_tr, y[train_idx])
        prob = clf.predict_proba(X_te)[:, 1]
        pred = clf.predict(X_te)
        aurocs.append(roc_auc_score(y[test_idx], prob))
        accs.append(accuracy_score(y[test_idx], pred))
    return aurocs, accs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, fe = load_whisper_encoder(device)

    results = {}

    # Load train metadata
    train_meta = pd.read_csv(TAUKADIAL_DIR / "train" / "groundtruth.csv")
    train_meta.rename(columns={train_meta.columns[0]: "tkdname"}, inplace=True)
    train_pids = train_meta["tkdname"].str.rsplit("-", n=1).str[0]
    logger.info("Train: %d recordings, %d participants, %d MCI",
                len(train_meta), train_pids.nunique(),
                (train_meta.groupby(train_pids)["dx"].first() == "MCI").sum())

    # Load test metadata
    test_gt = pd.read_csv(TAUKADIAL_DIR.parent / "testgroundtruth.csv", sep=";")
    test_gt.rename(columns={test_gt.columns[0]: "tkdname"}, inplace=True)
    test_pids = test_gt["tkdname"].str.rsplit("-", n=1).str[0]
    logger.info("Test: %d recordings, %d participants, %d MCI",
                len(test_gt), test_pids.nunique(),
                (test_gt.groupby(test_pids)["dx"].first() == "MCI").sum())

    # Extract embeddings
    logger.info("\nExtracting train embeddings...")
    train_embs = extract_embeddings(encoder, fe, TAUKADIAL_DIR / "train", train_meta, device)
    logger.info("Got %d train participants", len(train_embs))

    logger.info("\nExtracting test embeddings...")
    test_embs = extract_embeddings(encoder, fe, TAUKADIAL_DIR / "test", test_gt, device)
    logger.info("Got %d test participants", len(test_embs))

    # --- Eval 1: Train on train, eval on test ---
    train_pid_list = sorted(train_embs.keys())
    test_pid_list = sorted(test_embs.keys())

    X_train = np.array([train_embs[p] for p in train_pid_list])
    y_train = np.array([int(train_meta[train_meta["tkdname"].str.startswith(p)]["dx"].iloc[0] == "MCI")
                        for p in train_pid_list])
    X_test = np.array([test_embs[p] for p in test_pid_list])
    y_test = np.array([int(test_gt[test_gt["tkdname"].str.startswith(p)]["dx"].iloc[0] == "MCI")
                       for p in test_pid_list])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_tr_s, y_train)
    prob = clf.predict_proba(X_te_s)[:, 1]
    auroc_test = roc_auc_score(y_test, prob)
    acc_test = accuracy_score(y_test, clf.predict(X_te_s))
    logger.info("\n=== Train→Test: AUROC=%.3f, Acc=%.3f ===", auroc_test, acc_test)
    results["train_test"] = {"auroc": float(auroc_test), "accuracy": float(acc_test),
                             "n_train": len(train_pid_list), "n_test": len(test_pid_list)}

    # --- Eval 2: 5-fold CV on combined ---
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    aurocs, accs = run_cv(X_all, y_all)
    logger.info("=== 5-fold CV (all): AUROC=%.3f +/- %.3f, Acc=%.3f +/- %.3f ===",
                np.mean(aurocs), np.std(aurocs), np.mean(accs), np.std(accs))
    results["cv_all"] = {"auroc_mean": float(np.mean(aurocs)), "auroc_std": float(np.std(aurocs)),
                         "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                         "n": len(y_all)}

    # --- Eval 3: 5-fold CV on test only (40 participants) ---
    aurocs_test, accs_test = run_cv(X_test, y_test)
    logger.info("=== 5-fold CV (test only): AUROC=%.3f +/- %.3f ===",
                np.mean(aurocs_test), np.std(aurocs_test))
    results["cv_test"] = {"auroc_mean": float(np.mean(aurocs_test)), "auroc_std": float(np.std(aurocs_test)),
                          "n": len(y_test)}

    # Save
    out_path = PROJECT / "results_v3" / "taukadial_whisper_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nSaved to %s", out_path)


if __name__ == "__main__":
    main()
