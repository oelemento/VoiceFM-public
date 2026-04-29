#!/usr/bin/env python3
"""Frozen Whisper large-v2 baseline on GSD classification.

Extracts mean-pooled encoder embeddings (1280d) from vanilla Whisper
(no VoiceFM training) and runs logistic regression probes, using the
canonical create_participant_splits() for identical train/test sets as H7.

Also runs PCA-reduced 256d probes for dimensionality-matched comparison.

Usage:
    python scripts/whisper_baseline_gsd.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from src.data.sampler import create_participant_splits
from src.utils.preprocessing import load_and_preprocess, MAX_SAMPLES

LABELS = ["is_control_participant", "cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]
SEEDS = list(range(42, 47))
BATCH_SIZE = 16


def load_whisper_encoder(device, model_name="openai/whisper-large-v2"):
    """Load frozen Whisper encoder."""
    from transformers import WhisperModel, WhisperFeatureExtractor

    print(f"Loading {model_name}...")
    model = WhisperModel.from_pretrained(model_name, torch_dtype=torch.float32)
    encoder = model.encoder.float().to(device).eval()
    del model.decoder

    for param in encoder.parameters():
        param.requires_grad = False

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    print(f"  Encoder loaded ({sum(p.numel() for p in encoder.parameters()) / 1e6:.0f}M params)")
    return encoder, feature_extractor


def extract_embeddings(encoder, feature_extractor, recordings, device):
    """Extract mean-pooled 1280d embeddings per participant.

    Streams batches from disk. Uses masked mean-pooling to exclude
    padding tokens for short recordings.
    """
    audio_dir = PROJECT / "data" / "audio"

    # Build list of (pid, audio_path, wav_length) without loading audio
    items = []
    for _, row in recordings.iterrows():
        audio_path = audio_dir / row["audio_filename"]
        if audio_path.exists():
            items.append((row["record_id"], audio_path))

    print(f"    {len(items)} valid recordings, running batched inference...")
    pid_embeddings = {}

    for batch_start in range(0, len(items), BATCH_SIZE):
        if batch_start % 2000 == 0:
            print(f"    Processing {batch_start}/{len(items)}...")

        batch_items = items[batch_start:batch_start + BATCH_SIZE]

        # Load and preprocess this batch
        batch_wavs = []
        batch_pids = []
        batch_lengths = []
        for pid, audio_path in batch_items:
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
            batch_lengths.append(len(wav))

        if not batch_wavs:
            continue

        # Convert to mel features individually (handles variable lengths)
        mel_list = []
        for w in batch_wavs:
            mel_list.append(
                feature_extractor(
                    w, sampling_rate=16000, return_tensors="pt",
                ).input_features.squeeze(0)
            )
        mel = torch.stack(mel_list).to(device)  # (B, 80, 3000)

        with torch.no_grad():
            out = encoder(mel.float(), return_dict=True)
            hidden = out.last_hidden_state  # (B, 1500, 1280)

        # Masked mean-pooling: exclude padding tokens
        # Whisper encoder: 2 conv layers stride 2 → 4x downsample on mel frames
        # 30s = 3000 mel frames → 1500 tokens. Scale proportionally.
        max_samples = MAX_SAMPLES  # 30s * 16000 = 480000
        for i, (pid, length) in enumerate(zip(batch_pids, batch_lengths)):
            token_len = max(1, int(length / max_samples * hidden.shape[1]))
            token_len = min(token_len, hidden.shape[1])
            pooled = hidden[i, :token_len, :].mean(dim=0)  # (1280,)

            if pid not in pid_embeddings:
                pid_embeddings[pid] = []
            pid_embeddings[pid].append(pooled.cpu().numpy())

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/whisper-large-v2",
                        help="HuggingFace model ID (e.g. openai/whisper-large-v3)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    participants = pd.read_parquet(PROJECT / "data" / "processed" / "participants.parquet")
    recordings = pd.read_parquet(PROJECT / "data" / "processed" / "recordings.parquet")

    encoder, feature_extractor = load_whisper_encoder(device, model_name=args.model)

    # Extract ALL embeddings once
    print("\nExtracting embeddings for all recordings...")
    all_embs = extract_embeddings(encoder, feature_extractor, recordings, device)
    print(f"Got embeddings for {len(all_embs)} participants")

    results = {}

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")

        # Use canonical splits (same as H7 training)
        train_ids, val_ids, test_ids = create_participant_splits(
            participants,
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=seed, stratify_col="disease_category",
        )

        test_avail = [p for p in test_ids if p in all_embs]
        train_avail = [p for p in train_ids if p in all_embs]

        X_test = np.array([all_embs[p] for p in test_avail])
        X_train = np.array([all_embs[p] for p in train_avail])
        test_df = participants.loc[test_avail]
        train_df = participants.loc[train_avail]

        print(f"  train={len(train_avail)}, test={len(test_avail)}, dim={X_test.shape[1]}")

        # Full 1280d probes
        for label in LABELS:
            y_test = test_df[label].values.astype(int)
            y_train = train_df[label].values.astype(int)
            auroc = run_probe(X_train, y_train, X_test, y_test)

            key = f"whisper_frozen_1280d/{label}"
            if key not in results:
                results[key] = []
            results[key].append(auroc)
            print(f"  1280d {label}: {auroc:.3f}")

        # PCA to 256d for dimensionality-matched comparison
        pca = PCA(n_components=256, random_state=seed)
        X_train_256 = pca.fit_transform(StandardScaler().fit_transform(X_train))
        X_test_256 = pca.transform(StandardScaler().fit(X_train).transform(X_test))

        for label in LABELS:
            y_test = test_df[label].values.astype(int)
            y_train = train_df[label].values.astype(int)
            auroc = run_probe(X_train_256, y_train, X_test_256, y_test)

            key = f"whisper_frozen_256d/{label}"
            if key not in results:
                results[key] = []
            results[key].append(auroc)
            print(f"  256d  {label}: {auroc:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("Frozen Whisper Baseline (5-seed)")
    print("=" * 60)
    print(f"{'Category':<25} {'1280d':>14} {'256d (PCA)':>14}")
    print("-" * 55)
    for label in LABELS:
        v1280 = results.get(f"whisper_frozen_1280d/{label}", [])
        v256 = results.get(f"whisper_frozen_256d/{label}", [])
        s1280 = f"{np.mean(v1280):.3f}±{np.std(v1280):.3f}" if v1280 else "N/A"
        s256 = f"{np.mean(v256):.3f}±{np.std(v256):.3f}" if v256 else "N/A"
        print(f"  {label:<23} {s1280:>14} {s256:>14}")

    model_tag = args.model.replace("/", "_").replace("-", "_")
    out_path = PROJECT / "results" / f"whisper_baseline_gsd_{model_tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
