#!/usr/bin/env python3
"""Unified GSD probes: extract embeddings + run category & per-diagnosis probes.

Processes all 4 models with identical methodology:
  - VoiceFM-Whisper (5 seeds): checkpoints_exp_whisper_ft4_gsd_v3_seed{42-46}
  - VoiceFM-HuBERT (5 seeds): checkpoints_exp_d_gsd_v3_seed{42-46}
  - Frozen Whisper (1 set): from whisper_recording_embeddings.npz
  - Frozen HuBERT (1 set): extracted once with frozen weights

Each seed: extract embeddings → mean-pool per participant →
create_participant_splits(seed) → StandardScaler → LogisticRegression.

Output: results_v3/unified_gsd_probes.json

Usage:
    python scripts/unified_gsd_probes_v3.py [--seeds 42 43 44 45 46]
"""

import argparse
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
                    format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).parent.parent
RESULTS = PROJECT / "results_v3"
BATCH_SIZE = 8  # Must match reference script — Whisper time-masking depends on batch max_len
SEEDS = [42, 43, 44, 45, 46]

GSD_CATS = ["gsd_control", "cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]
GSD_DIAGS = [
    "gsd_parkinsons", "gsd_alz_dementia_mci", "gsd_mtd", "gsd_copd_asthma",
    "gsd_depression", "gsd_airway_stenosis", "gsd_benign_lesion", "gsd_anxiety",
    "gsd_laryngeal_dystonia",
]
ALL_LABELS = GSD_CATS + GSD_DIAGS


# ── Model loading ────────────────────────────────────────────────────────

def load_whisper_encoder(seed, device):
    ckpt_path = PROJECT / f"checkpoints_exp_whisper_ft4_gsd_v3_seed{seed}" / "best_model.pt"
    if not ckpt_path.exists():
        return None
    with open(PROJECT / "configs" / "model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    model_cfg["audio_encoder"]["type"] = "whisper"
    model_cfg["audio_encoder"]["backbone"] = "openai/whisper-large-v2"
    model_cfg["audio_encoder"]["freeze_backbone"] = True
    model_cfg["audio_encoder"]["unfreeze_last_n"] = 4

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    num_tt = state["audio_encoder.task_embedding.weight"].shape[0] if "audio_encoder.task_embedding.weight" in state else 100
    encoder = build_audio_encoder(config=model_cfg["audio_encoder"], num_task_types=num_tt)
    ae_state = {k.replace("audio_encoder.", "", 1): v for k, v in state.items() if k.startswith("audio_encoder.")}
    encoder.load_state_dict(ae_state)
    return encoder.to(device).eval()


def load_hubert_encoder(seed, device):
    ckpt_path = PROJECT / f"checkpoints_exp_d_gsd_v3_seed{seed}" / "best_model.pt"
    if not ckpt_path.exists():
        return None
    with open(PROJECT / "configs" / "model.yaml") as f:
        model_cfg = yaml.safe_load(f)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    num_tt = state["audio_encoder.task_embedding.weight"].shape[0] if "audio_encoder.task_embedding.weight" in state else 100
    encoder = build_audio_encoder(config=model_cfg["audio_encoder"], num_task_types=num_tt)
    ae_state = {k.replace("audio_encoder.", "", 1): v for k, v in state.items() if k.startswith("audio_encoder.")}
    encoder.load_state_dict(ae_state)
    return encoder.to(device).eval()


def load_frozen_hubert_encoder(device):
    with open(PROJECT / "configs" / "model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    cfg = model_cfg["audio_encoder"].copy()
    cfg["freeze_backbone"] = True
    cfg["unfreeze_last_n"] = 0
    torch.manual_seed(42)  # Seed random projection weights for reproducibility
    encoder = build_audio_encoder(config=cfg, num_task_types=100)
    return encoder.to(device).eval()


# ── Embedding extraction ─────────────────────────────────────────────────

@torch.no_grad()
def extract_participant_embeddings(encoder, recordings, device, mask_dtype=np.float32):
    """Extract embeddings, mean-pool per participant.

    Uses proper task_type_ids from build_task_type_map.
    mask_dtype: np.int64 for Whisper encoders, np.float32 for HuBERT encoders.
    """
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
        masks = np.zeros((len(wavs), max_len), dtype=mask_dtype)
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

    pid_mean = {pid: np.mean(emb_list, axis=0) for pid, emb_list in pid_embs.items()}
    return pid_mean


def load_frozen_whisper_embeddings():
    """Load pre-extracted frozen Whisper recordings, mean-pool per participant."""
    path = RESULTS / "whisper_recording_embeddings.npz"
    data = np.load(path, allow_pickle=True)
    pid_idx = {}
    for i, pid in enumerate(data["participant_ids"]):
        pid_idx.setdefault(str(pid), []).append(i)
    return {pid: data["embeddings"][idxs].mean(axis=0) for pid, idxs in pid_idx.items()}


# ── Probe evaluation ─────────────────────────────────────────────────────

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
    """Run category + per-diagnosis probes with create_participant_splits."""
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


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

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
    logger.info("Recordings: %d, Participants: %d", len(recordings), len(participants))

    all_results = {}
    out_path = RESULTS / "unified_gsd_probes.json"

    def save_incremental():
        """Save after each model so crashes don't lose earlier results."""
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("  [incremental save: %d keys]", len(all_results))

    # ── Frozen Whisper (from pre-extracted npz) ──────────────────────
    logger.info("\n=== Frozen Whisper 1280d ===")
    fw_embs = load_frozen_whisper_embeddings()
    logger.info("Loaded %d participants", len(fw_embs))
    for seed in args.seeds:
        probes = run_all_probes(fw_embs, participants, seed)
        for k, v in probes.items():
            all_results.setdefault(f"frozen_whisper/{k}", []).append(v)
        cats = [probes.get(c, float("nan")) for c in GSD_CATS]
        logger.info("  seed %d: mean=%.3f [%s]", seed, np.nanmean(cats),
                    " ".join(f"{v:.3f}" for v in cats))
    save_incremental()

    # ── Trained models (per-seed embeddings) — run BEFORE frozen HuBERT
    #    so that a crash in frozen HuBERT doesn't block VoiceFM results ──
    for model_name, loader_fn, mdtype in [
        ("voicefm_whisper", load_whisper_encoder, np.int64),    # Whisper uses int64 masks
        ("voicefm_hubert", load_hubert_encoder, np.float32),    # HuBERT uses float32 masks
    ]:
        logger.info("\n=== %s (5 seeds, mask_dtype=%s) ===", model_name, mdtype.__name__)
        for seed in args.seeds:
            encoder = loader_fn(seed, device)
            if encoder is None:
                logger.warning("  Seed %d: checkpoint not found, skipping", seed)
                continue
            pid_mean = extract_participant_embeddings(encoder, recordings, device, mask_dtype=mdtype)
            logger.info("  Seed %d: %d participants (%dd)", seed, len(pid_mean),
                        len(next(iter(pid_mean.values()))))
            del encoder; torch.cuda.empty_cache()

            # Save embeddings (for local figure generation)
            pids_s = sorted(pid_mean.keys())
            np.savez_compressed(
                RESULTS / f"{model_name}_seed{seed}_embeddings.npz",
                pids=np.array(pids_s),
                embeddings=np.array([pid_mean[p] for p in pids_s]))

            probes = run_all_probes(pid_mean, participants, seed)
            for k, v in probes.items():
                all_results.setdefault(f"{model_name}/{k}", []).append(v)
            cats = [probes.get(c, float("nan")) for c in GSD_CATS]
            logger.info("    mean=%.3f [%s]", np.nanmean(cats),
                        " ".join(f"{v:.3f}" for v in cats))
        save_incremental()

    # ── Frozen HuBERT (last — crashes don't block VoiceFM results) ───
    logger.info("\n=== Frozen HuBERT ===")
    try:
        fh_npz = RESULTS / "frozen_hubert_embeddings.npz"
        if fh_npz.exists():
            logger.info("Loading from cached %s", fh_npz)
            data = np.load(fh_npz, allow_pickle=True)
            fh_embs = {str(pid): emb for pid, emb in zip(data["pids"], data["embeddings"])}
        else:
            fh_encoder = load_frozen_hubert_encoder(device)
            fh_embs = extract_participant_embeddings(fh_encoder, recordings, device)
            del fh_encoder; torch.cuda.empty_cache()
            pids_s = sorted(fh_embs.keys())
            np.savez_compressed(fh_npz,
                                pids=np.array(pids_s),
                                embeddings=np.array([fh_embs[p] for p in pids_s]))
        logger.info("Frozen HuBERT: %d participants (%dd)",
                    len(fh_embs), len(next(iter(fh_embs.values()))))

        for seed in args.seeds:
            probes = run_all_probes(fh_embs, participants, seed)
            for k, v in probes.items():
                all_results.setdefault(f"frozen_hubert/{k}", []).append(v)
            cats = [probes.get(c, float("nan")) for c in GSD_CATS]
            logger.info("  seed %d: mean=%.3f [%s]", seed, np.nanmean(cats),
                        " ".join(f"{v:.3f}" for v in cats))
        save_incremental()
    except Exception as e:
        logger.error("Frozen HuBERT failed: %s", e)
        logger.info("Continuing without frozen HuBERT results")

    # ── Final save ───────────────────────────────────────────────────
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nSaved to %s", out_path)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for model in ["voicefm_whisper", "voicefm_hubert", "frozen_whisper", "frozen_hubert"]:
        cats = [np.mean(all_results.get(f"{model}/{c}", [])) for c in GSD_CATS]
        diags = [np.mean(all_results.get(f"{model}/{d}", [])) for d in GSD_DIAGS
                 if all_results.get(f"{model}/{d}")]
        logger.info("%-20s  cat_mean=%.3f  diag_mean=%.3f  n_seeds=%d",
                    model, np.nanmean(cats), np.nanmean(diags),
                    len(all_results.get(f"{model}/gsd_control", [])))


if __name__ == "__main__":
    main()
