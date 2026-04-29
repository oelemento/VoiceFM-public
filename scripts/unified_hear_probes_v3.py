#!/usr/bin/env python3
"""Unified HeAR probes: Frozen HeAR + HeAR VoiceFM evaluation.

Same methodology as unified_gsd_probes_v3.py:
  - Extract embeddings, mean-pool per participant
  - create_participant_splits(seed) → StandardScaler → LogisticRegression
  - 5 seeds (42-46)

Models:
  - Frozen HeAR: raw 512d HeAR pooler output, no training
  - HeAR VoiceFM: contrastive-trained 256d (frozen HeAR backbone + projection)

Output: updates results_v3/unified_gsd_probes.json with hear_voicefm/* and frozen_hear/* keys.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sampler import create_participant_splits
from src.data.audio_dataset import build_task_type_map
from src.models import build_audio_encoder
from src.utils.preprocessing import load_and_preprocess, MAX_SAMPLES

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).parent.parent
RESULTS = PROJECT / "results_v3"
BATCH_SIZE = 8  # Match unified_gsd_probes.py
SEEDS = [42, 43, 44, 45, 46]

GSD_CATS = ["gsd_control", "cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]
GSD_DIAGS = [
    "gsd_parkinsons", "gsd_alz_dementia_mci", "gsd_mtd", "gsd_copd_asthma",
    "gsd_depression", "gsd_airway_stenosis", "gsd_benign_lesion", "gsd_anxiety",
    "gsd_laryngeal_dystonia",
]
ALL_LABELS = GSD_CATS + GSD_DIAGS


@torch.no_grad()
def extract_frozen_hear_embeddings(recordings, device):
    """Extract raw 512d HeAR embeddings (frozen, no VoiceFM projection)."""
    from transformers import AutoModel
    import torch.nn.functional as F

    # Load HeAR model directly
    hear = AutoModel.from_pretrained("google/hear-pytorch", trust_remote_code=True)
    hear = hear.to(device).eval()

    # HeAR preprocessor
    from src.models.hear_encoder import MelPCENPreprocessor, HEAR_CHUNK_SAMPLES, MIN_CHUNK_SAMPLES
    preprocessor = MelPCENPreprocessor().to(device)

    audio_dir = PROJECT / "data" / "audio"
    items = [(row["record_id"], audio_dir / row["audio_filename"])
             for _, row in recordings.iterrows()
             if (audio_dir / row["audio_filename"]).exists()]

    pid_embs = {}
    n_errors = 0
    for i, (pid, path) in enumerate(items):
        try:
            wav = load_and_preprocess(str(path), max_samples=MAX_SAMPLES)
            if isinstance(wav, torch.Tensor):
                wav = wav.numpy()
            if len(wav) < 400:
                continue

            audio_t = torch.tensor(wav, dtype=torch.float32, device=device).unsqueeze(0)
            # Chunk into 2-second segments
            n_samples = audio_t.shape[1]
            chunks = []
            for start in range(0, n_samples, HEAR_CHUNK_SAMPLES):
                chunk = audio_t[:, start:start + HEAR_CHUNK_SAMPLES]
                if chunk.shape[1] >= MIN_CHUNK_SAMPLES:
                    # Pad to 2 seconds if needed
                    if chunk.shape[1] < HEAR_CHUNK_SAMPLES:
                        chunk = F.pad(chunk, (0, HEAR_CHUNK_SAMPLES - chunk.shape[1]))
                    chunks.append(chunk.squeeze(0))

            if not chunks:
                continue

            chunk_batch = torch.stack(chunks).to(device)
            # Preprocess (mel-PCEN) and run HeAR
            with torch.amp.autocast(device_type="cuda", enabled=False):
                spectrograms = preprocessor(chunk_batch.float())
                out = hear(pixel_values=spectrograms, return_dict=True)
                chunk_embs = out.pooler_output  # (n_chunks, 512)

            # Mean-pool chunks, L2-normalize
            pooled = chunk_embs.mean(dim=0)
            pooled = F.normalize(pooled, p=2, dim=-1)
            pid_embs.setdefault(pid, []).append(pooled.cpu().numpy())

        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                logger.warning("Error on recording %d (%s): %s", i, path.name, e)

        if i % 5000 == 0 and i > 0:
            logger.info("  %d/%d recordings (%d errors)...", i, len(items), n_errors)

    del hear, preprocessor
    torch.cuda.empty_cache()
    logger.info("Frozen HeAR: %d participants (%d errors / %d recordings)",
                len(pid_embs), n_errors, len(items))
    return {pid: np.mean(emb_list, axis=0) for pid, emb_list in pid_embs.items()}


@torch.no_grad()
def extract_hear_voicefm_embeddings(seed, recordings, device):
    """Extract 256d embeddings from HeAR VoiceFM checkpoint."""
    ckpt_path = PROJECT / f"checkpoints_exp_hear_gsd_v3_seed{seed}" / "best_model.pt"
    if not ckpt_path.exists():
        return None

    with open(PROJECT / "configs" / "model.yaml") as f:
        model_cfg = yaml.safe_load(f)

    # Override to HeAR type
    cfg = model_cfg["audio_encoder"].copy()
    cfg["type"] = "hear"
    cfg["backbone"] = "google/hear-pytorch"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    task_emb_key = "audio_encoder.task_embedding.weight"
    num_tt = state[task_emb_key].shape[0] if task_emb_key in state else 100

    encoder = build_audio_encoder(config=cfg, num_task_types=num_tt)
    ae_state = {k.replace("audio_encoder.", "", 1): v
                for k, v in state.items() if k.startswith("audio_encoder.")}
    encoder.load_state_dict(ae_state)
    encoder = encoder.to(device).eval()

    audio_dir = PROJECT / "data" / "audio"
    task_type_map = build_task_type_map(recordings)
    items = [(row["record_id"], row.get("recording_name", ""),
              audio_dir / row["audio_filename"])
             for _, row in recordings.iterrows()
             if (audio_dir / row["audio_filename"]).exists()]

    pid_embs = {}
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        wavs, pids, task_ids_list = [], [], []
        for pid, task_name, path in batch:
            try:
                wav = load_and_preprocess(str(path), max_samples=MAX_SAMPLES)
            except Exception:
                continue
            if isinstance(wav, torch.Tensor):
                wav = wav.numpy()
            if len(wav) < 400:
                continue
            wavs.append(wav)
            pids.append(pid)
            task_ids_list.append(task_type_map.get(task_name, 0))
        if not wavs:
            continue

        max_len = max(len(w) for w in wavs)
        padded = np.zeros((len(wavs), max_len), dtype=np.float32)
        masks = np.zeros((len(wavs), max_len), dtype=np.float32)
        for j, w in enumerate(wavs):
            padded[j, :len(w)] = w
            masks[j, :len(w)] = 1

        embs = encoder(
            torch.tensor(padded, device=device),
            torch.tensor(masks, device=device),
            torch.tensor(task_ids_list, dtype=torch.long, device=device),
        ).cpu().numpy()

        for pid, emb in zip(pids, embs):
            pid_embs.setdefault(pid, []).append(emb)

        if i % 4000 == 0 and i > 0:
            logger.info("  %d/%d recordings...", i, len(items))

    del encoder
    torch.cuda.empty_cache()
    return {pid: np.mean(emb_list, axis=0) for pid, emb_list in pid_embs.items()}


def run_probe(X_train, y_train, X_test, y_test):
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return float("nan")
    if np.sum(y_test) < 2 or np.sum(y_train) < 2:
        return float("nan")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_tr_s, y_train)
    probs = clf.predict_proba(X_te_s)[:, 1]
    return float(roc_auc_score(y_test, probs))


def run_all_probes(pid_mean, participants, seed):
    train_ids, _, test_ids = create_participant_splits(
        participants, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
        seed=seed, stratify_col="disease_category",
    )
    train_avail = [p for p in train_ids if p in pid_mean]
    test_avail = [p for p in test_ids if p in pid_mean]
    X_train = np.array([pid_mean[p] for p in train_avail])
    X_test = np.array([pid_mean[p] for p in test_avail])
    train_df = participants.loc[train_avail]
    test_df = participants.loc[test_avail]

    results = {}
    for label in ALL_LABELS:
        if label not in participants.columns:
            continue
        y_tr = train_df[label].values.astype(int)
        y_te = test_df[label].values.astype(int)
        auroc = run_probe(X_train, y_tr, X_test, y_te)
        if not np.isnan(auroc):
            results[label] = auroc
    return results


def save_incremental(all_results, out_path):
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("  [saved: %d keys]", len(all_results))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    recordings = pd.read_parquet(PROJECT / "data" / "processed_v3" / "recordings.parquet")
    participants = pd.read_parquet(PROJECT / "data" / "processed_v3" / "participants.parquet")
    if "participant_id" in participants.columns:
        participants = participants.set_index("participant_id")
    # v3: restrict to the training cohort (846 participants).
    # The 138 'test' (prospective) participants are evaluated separately.
    if "cohort_split" in participants.columns:
        before = len(participants)
        participants = participants[participants["cohort_split"] == "train"].copy()
        logger.info("Filtered to cohort_split==train: %d → %d participants", before, len(participants))

    # Load existing unified results to merge into
    out_path = RESULTS / "unified_gsd_probes.json"
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Remove old HeAR keys
    all_results = {k: v for k, v in all_results.items()
                   if not k.startswith("frozen_hear/") and not k.startswith("hear_voicefm/")}

    # ── Frozen HeAR (extract once, probe with 5 seeds) ────────────────
    logger.info("\n=== Frozen HeAR 512d ===")
    fh_embs = extract_frozen_hear_embeddings(recordings, device)
    logger.info("Extracted %d participants (%dd)",
                len(fh_embs), len(next(iter(fh_embs.values()))))

    for seed in SEEDS:
        probes = run_all_probes(fh_embs, participants, seed)
        for k, v in probes.items():
            all_results.setdefault(f"frozen_hear/{k}", []).append(v)
        cats = [probes.get(c, float("nan")) for c in GSD_CATS]
        logger.info("  seed %d: mean=%.3f [%s]", seed, np.nanmean(cats),
                    " ".join(f"{v:.3f}" for v in cats))
    save_incremental(all_results, out_path)

    # ── HeAR VoiceFM (per-seed extraction + probes) ───────────────────
    logger.info("\n=== HeAR VoiceFM (5 seeds) ===")
    for seed in SEEDS:
        pid_mean = extract_hear_voicefm_embeddings(seed, recordings, device)
        if pid_mean is None:
            logger.warning("  Seed %d: checkpoint not found, skipping", seed)
            continue
        logger.info("  Seed %d: %d participants (%dd)", seed, len(pid_mean),
                    len(next(iter(pid_mean.values()))))

        probes = run_all_probes(pid_mean, participants, seed)
        for k, v in probes.items():
            all_results.setdefault(f"hear_voicefm/{k}", []).append(v)
        cats = [probes.get(c, float("nan")) for c in GSD_CATS]
        logger.info("    mean=%.3f [%s]", np.nanmean(cats),
                    " ".join(f"{v:.3f}" for v in cats))
    save_incremental(all_results, out_path)

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    for model in ["frozen_hear", "hear_voicefm"]:
        cats = [np.mean(all_results.get(f"{model}/{c}", [])) for c in GSD_CATS]
        n = len(all_results.get(f"{model}/gsd_control", []))
        logger.info("%-20s  cat_mean=%.3f  n_seeds=%d", model, np.nanmean(cats), n)


if __name__ == "__main__":
    main()
