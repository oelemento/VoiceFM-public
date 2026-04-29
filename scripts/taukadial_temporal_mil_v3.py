#!/usr/bin/env python3
"""TAUKADIAL MCI detection via temporal Multiple Instance Learning.

Approach:
1. Extract per-segment (5s window) Whisper hidden states from B2AI recordings
2. Train segment-level neuro classifier on B2AI (AD/MCI/neuro vs controls)
3. Apply to TAUKADIAL recordings → P(impairment) per segment
4. Aggregate per participant: max, percentile, variance, fraction-above-threshold
5. Classify MCI vs NC using aggregated temporal features

Usage:
    python -u scripts/taukadial_temporal_mil_v3.py
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from src.data.sampler import create_participant_splits
from src.utils.preprocessing import load_and_preprocess, MAX_SAMPLES

SEGMENT_SECONDS = 5
SEGMENT_SAMPLES = 16000 * SEGMENT_SECONDS  # 80000 samples per segment

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_whisper(device):
    from transformers import WhisperModel, WhisperFeatureExtractor
    logger.info("Loading Whisper large-v2...")
    model = WhisperModel.from_pretrained("openai/whisper-large-v2", torch_dtype=torch.float32)
    encoder = model.encoder.float().to(device).eval()
    del model.decoder
    for p in encoder.parameters():
        p.requires_grad = False
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")
    return encoder, fe


def extract_segment_embeddings(encoder, fe, audio_path, device):
    """Extract per-segment (5s) Whisper embeddings from a single recording.

    Returns list of 1280d numpy arrays, one per segment.
    """
    try:
        wav = load_and_preprocess(str(audio_path), max_samples=int(1e9))  # keep full length
    except Exception:
        return []

    if isinstance(wav, torch.Tensor):
        wav = wav.numpy()
    if len(wav) < SEGMENT_SAMPLES // 2:
        return []

    # Split into 5-second segments
    segments = []
    for start in range(0, len(wav), SEGMENT_SAMPLES):
        seg = wav[start:start + SEGMENT_SAMPLES]
        if len(seg) < SEGMENT_SAMPLES // 2:  # skip very short tail
            continue
        segments.append(seg.astype(np.float32))

    if not segments:
        return []

    # Process all segments through Whisper (each padded to 30s by feature extractor)
    segment_embs = []
    for seg in segments:
        mel = fe(seg, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            out = encoder(mel.float(), return_dict=True)
            hidden = out.last_hidden_state  # (1, 1500, 1280)
            # Pool only over tokens corresponding to actual audio
            token_len = max(1, int(len(seg) / MAX_SAMPLES * hidden.shape[1]))
            token_len = min(token_len, hidden.shape[1])
            pooled = hidden[0, :token_len, :].mean(dim=0).cpu().numpy()
            segment_embs.append(pooled)

    return segment_embs


def train_b2ai_segment_classifier(encoder, fe, device):
    """Train segment-level neuro classifier on B2AI data.

    Uses gsd_alz_dementia_mci + other neuro conditions as positive,
    gsd_control as negative.
    """
    participants = pd.read_parquet(PROJECT / "data" / "processed_v3" / "participants.parquet")
    recordings = pd.read_parquet(PROJECT / "data" / "processed_v3" / "recordings.parquet")
    audio_dir = PROJECT / "data" / "audio"

    # Select neuro (positive) and control (negative) participants
    neuro_pids = participants[participants["cat_neuro"] == 1].index.tolist()
    control_pids = participants[participants["gsd_control"] == 1].index.tolist()

    logger.info("B2AI calibration: %d neuro, %d control participants", len(neuro_pids), len(control_pids))

    # Use a subset of recordings (max 10 per participant to keep it manageable)
    all_segments = []
    all_labels = []

    for label, pids in [(1, neuro_pids), (0, control_pids)]:
        pid_recs = recordings[recordings["record_id"].isin(pids)]
        # Sample max 10 recordings per participant
        for pid in pids:
            pid_subset = pid_recs[pid_recs["record_id"] == pid].head(10)
            for _, row in pid_subset.iterrows():
                audio_path = audio_dir / row["audio_filename"]
                if not audio_path.exists():
                    continue
                segs = extract_segment_embeddings(encoder, fe, audio_path, device)
                for seg in segs:
                    all_segments.append(seg)
                    all_labels.append(label)

            if len(all_segments) % 500 == 0:
                logger.info("  Extracted %d segments so far...", len(all_segments))

    X = np.array(all_segments)
    y = np.array(all_labels)
    logger.info("B2AI segments: %d total (%d neuro, %d control)", len(y), y.sum(), (1-y).sum())

    # Train logistic regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=0.1, random_state=42, class_weight="balanced")
    clf.fit(X_scaled, y)

    # Quick sanity check: train accuracy
    train_prob = clf.predict_proba(X_scaled)[:, 1]
    train_auroc = roc_auc_score(y, train_prob)
    logger.info("B2AI segment classifier train AUROC: %.3f", train_auroc)

    return clf, scaler


def score_taukadial(encoder, fe, clf, scaler, audio_dir, metadata, device):
    """Apply B2AI-trained segment classifier to TAUKADIAL recordings.

    Returns per-participant aggregated features.
    """
    pid_segments = {}  # pid -> list of P(impairment) scores

    for idx, (_, row) in enumerate(metadata.iterrows()):
        fname = row["tkdname"] if "tkdname" in row else row.iloc[0]
        pid = fname.replace(".wav", "").rsplit("-", 1)[0]
        wav_path = audio_dir / fname

        if not wav_path.exists():
            logger.warning("Missing: %s", fname)
            continue

        wav, sr = sf.read(wav_path)
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            gcd = math.gcd(sr, 16000)
            wav = resample_poly(wav, 16000 // gcd, sr // gcd).astype(np.float32)

        # Segment and extract
        segments = []
        for start in range(0, len(wav), SEGMENT_SAMPLES):
            seg = wav[start:start + SEGMENT_SAMPLES]
            if len(seg) < SEGMENT_SAMPLES // 2:
                continue
            segments.append(seg.astype(np.float32))

        if not segments:
            continue

        # Get embeddings and score
        seg_embs = []
        for seg in segments:
            mel = fe(seg, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            with torch.no_grad():
                out = encoder(mel.float(), return_dict=True)
                token_len = max(1, int(len(seg) / MAX_SAMPLES * out.last_hidden_state.shape[1]))
                pooled = out.last_hidden_state[0, :token_len, :].mean(dim=0).cpu().numpy()
                seg_embs.append(pooled)

        X_seg = scaler.transform(np.array(seg_embs))
        scores = clf.predict_proba(X_seg)[:, 1]  # P(neuro impairment)

        if pid not in pid_segments:
            pid_segments[pid] = []
        pid_segments[pid].extend(scores.tolist())

        if (idx + 1) % 50 == 0:
            logger.info("  Scored %d/%d recordings...", idx + 1, len(metadata))

    # Aggregate per participant
    pid_features = {}
    for pid, scores in pid_segments.items():
        scores = np.array(scores)
        pid_features[pid] = {
            "mean": float(scores.mean()),
            "max": float(scores.max()),
            "p90": float(np.percentile(scores, 90)),
            "p95": float(np.percentile(scores, 95)),
            "std": float(scores.std()),
            "frac_above_50": float((scores > 0.5).mean()),
            "frac_above_70": float((scores > 0.7).mean()),
            "top3_mean": float(np.sort(scores)[-3:].mean()) if len(scores) >= 3 else float(scores.max()),
            "n_segments": len(scores),
            # Also keep the full embedding mean for fusion
        }

    return pid_features


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, fe = load_whisper(device)

    # Step 1: Train segment-level classifier on B2AI
    logger.info("\n=== Step 1: Training B2AI segment classifier ===")
    clf, scaler = train_b2ai_segment_classifier(encoder, fe, device)

    # Step 2: Score TAUKADIAL
    logger.info("\n=== Step 2: Scoring TAUKADIAL ===")
    taukadial_dir = PROJECT / "data" / "external" / "taukadial" / "TAUKADIAL-24"

    # Train set
    train_meta = pd.read_csv(taukadial_dir / "train" / "groundtruth.csv")
    train_meta.rename(columns={train_meta.columns[0]: "tkdname"}, inplace=True)
    logger.info("Scoring TAUKADIAL train (%d recordings)...", len(train_meta))
    train_features = score_taukadial(encoder, fe, clf, scaler, taukadial_dir / "train", train_meta, device)

    # Test set
    test_meta = pd.read_csv(taukadial_dir.parent / "testgroundtruth.csv", sep=";")
    test_meta.rename(columns={test_meta.columns[0]: "tkdname"}, inplace=True)
    logger.info("Scoring TAUKADIAL test (%d recordings)...", len(test_meta))
    test_features = score_taukadial(encoder, fe, clf, scaler, taukadial_dir / "test", test_meta, device)

    # Step 3: Classify MCI vs NC
    logger.info("\n=== Step 3: Classifying MCI vs NC ===")

    feature_names = ["mean", "max", "p90", "p95", "std", "frac_above_50", "frac_above_70", "top3_mean"]

    # Build feature matrices
    train_pids = sorted(train_features.keys())
    test_pids = sorted(test_features.keys())

    X_train = np.array([[train_features[p][f] for f in feature_names] for p in train_pids])
    X_test = np.array([[test_features[p][f] for f in feature_names] for p in test_pids])

    y_train = np.array([int(train_meta[train_meta["tkdname"].str.startswith(p)]["dx"].iloc[0] == "MCI")
                        for p in train_pids])
    y_test = np.array([int(test_meta[test_meta["tkdname"].str.startswith(p)]["dx"].iloc[0] == "MCI")
                       for p in test_pids])

    logger.info("Features: %s", feature_names)
    logger.info("Train: %d (%d MCI), Test: %d (%d MCI)",
                len(y_train), y_train.sum(), len(y_test), y_test.sum())

    # Evaluate: train→test
    scaler2 = StandardScaler()
    X_tr = scaler2.fit_transform(X_train)
    X_te = scaler2.transform(X_test)

    clf2 = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf2.fit(X_tr, y_train)
    prob = clf2.predict_proba(X_te)[:, 1]
    auroc = roc_auc_score(y_test, prob)
    logger.info("\n=== RESULTS ===")
    logger.info("Temporal MIL train→test AUROC: %.3f", auroc)

    # Also try individual features
    logger.info("\nPer-feature AUROCs:")
    for i, fname in enumerate(feature_names):
        feat_auroc = roc_auc_score(y_test, X_test[:, i])
        logger.info("  %s: %.3f", fname, feat_auroc)

    # 5-fold CV on combined
    from sklearn.model_selection import StratifiedKFold
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aurocs = []
    for tr_idx, te_idx in skf.split(X_all, y_all):
        sc = StandardScaler()
        c = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        c.fit(sc.fit_transform(X_all[tr_idx]), y_all[tr_idx])
        p = c.predict_proba(sc.transform(X_all[te_idx]))[:, 1]
        cv_aurocs.append(roc_auc_score(y_all[te_idx], p))
    logger.info("5-fold CV (all): AUROC=%.3f +/- %.3f", np.mean(cv_aurocs), np.std(cv_aurocs))

    # Compare with baseline (frozen Whisper mean-pool)
    logger.info("\nBaseline comparison (frozen Whisper mean-pool): 0.546 train→test, 0.663 CV")

    # Save
    results = {
        "temporal_mil": {
            "train_test_auroc": float(auroc),
            "cv_auroc_mean": float(np.mean(cv_aurocs)),
            "cv_auroc_std": float(np.std(cv_aurocs)),
            "per_feature_aurocs": {fname: float(roc_auc_score(y_test, X_test[:, i]))
                                   for i, fname in enumerate(feature_names)},
            "n_train": len(y_train),
            "n_test": len(y_test),
        },
        "baseline": {
            "train_test_auroc": 0.546,
            "cv_auroc": 0.663,
        }
    }

    out_path = PROJECT / "results_v3" / "taukadial_temporal_mil.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
