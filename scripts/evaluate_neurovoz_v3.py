#!/usr/bin/env python3
"""Evaluate VoiceFM and HuBERT on NeuroVoz PD detection (participant-level).

NeuroVoz: 108 participants (55 HC, 53 PD), Spanish, multiple task types.
All evaluation is participant-level: embeddings are mean-pooled per participant,
CV folds split by participant, few-shot samples participants (not recordings).

Evaluates:
  1. Per task category: all, vowel, speech, ddk
  2. Few-shot (k=1,2,5,10,20) for "all" and "speech"

Usage:
    python scripts/evaluate_neurovoz_v3.py --checkpoint checkpoints_exp_d_gsd_seed42/best_model.pt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import HubertModel

from src.data.external_datasets import NeuroVozDataset, external_collate_fn
from src.models import build_audio_encoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
NEUROVOZ_DIR = PROJECT_ROOT / "data" / "external" / "neurovoz"


def aggregate_to_participants(embeddings, labels, participant_ids):
    """Mean-pool embeddings per participant, return participant-level arrays."""
    pid_to_embs = {}
    pid_to_label = {}
    for emb, label, pid in zip(embeddings, labels, participant_ids):
        if pid not in pid_to_embs:
            pid_to_embs[pid] = []
            pid_to_label[pid] = label
        pid_to_embs[pid].append(emb)

    pids = sorted(pid_to_embs.keys())
    X = np.array([np.mean(pid_to_embs[p], axis=0) for p in pids])
    y = np.array([pid_to_label[p] for p in pids])
    return X, y, pids


def extract_voicefm_embeddings(audio_encoder, dataloader, device):
    """Extract 256d (post-projection) embeddings + participant_ids."""
    audio_encoder.eval()
    all_embs, all_labels, all_pids = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            audio = batch["audio_values"].to(device)
            mask = batch["attention_mask"].to(device)
            task_ids = torch.zeros(audio.shape[0], dtype=torch.long, device=device)

            embeds = audio_encoder(
                audio_input_values=audio, attention_mask=mask, task_type_ids=task_ids,
            )
            all_embs.append(embeds.cpu().numpy())
            all_labels.extend(batch["labels"]["disease"].numpy())
            all_pids.extend(batch["participant_ids"])

    return np.concatenate(all_embs), np.array(all_labels), all_pids


def extract_hubert_embeddings(hubert, dataloader, device):
    """Extract attention-mask-aware mean-pooled 768d embeddings from frozen HuBERT."""
    hubert.eval()
    all_embs, all_labels, all_pids = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            audio = batch["audio_values"].to(device)
            mask = batch["attention_mask"].to(device)
            out = hubert(audio, attention_mask=mask, return_dict=True)
            hidden = out.last_hidden_state
            frame_mask = hubert._get_feature_vector_attention_mask(hidden.shape[1], mask)
            frame_mask_f = frame_mask.unsqueeze(-1).float()
            pooled = (hidden * frame_mask_f).sum(dim=1) / frame_mask_f.sum(dim=1).clamp(min=1)
            pooled = F.normalize(pooled, p=2, dim=-1)
            all_embs.append(pooled.cpu().numpy())
            all_labels.extend(batch["labels"]["disease"].numpy())
            all_pids.extend(batch["participant_ids"])

    return np.concatenate(all_embs), np.array(all_labels), all_pids


def evaluate_probe(X, y, n_splits=5):
    """5-fold CV logistic regression AUROC on participant-level data."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aurocs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_tr, y[train_idx])
        prob = clf.predict_proba(X_te)[:, 1]
        aurocs.append(roc_auc_score(y[test_idx], prob))
    return float(np.mean(aurocs)), float(np.std(aurocs))


def few_shot_eval(X, y, k_values=(1, 2, 5, 10, 20), n_trials=100):
    """Few-shot evaluation: k participants per class."""
    results = {}
    for k in k_values:
        if k > min(np.sum(y == 0), np.sum(y == 1)):
            continue
        aurocs = []
        for trial in range(n_trials):
            rng = np.random.RandomState(trial)
            idx_0 = rng.choice(np.where(y == 0)[0], k, replace=False)
            idx_1 = rng.choice(np.where(y == 1)[0], k, replace=False)
            train_idx = np.concatenate([idx_0, idx_1])
            test_idx = np.setdiff1d(np.arange(len(y)), train_idx)

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X_tr, y[train_idx])
            prob = clf.predict_proba(X_te)[:, 1]
            try:
                aurocs.append(roc_auc_score(y[test_idx], prob))
            except ValueError:
                pass
        if aurocs:
            results[k] = {"mean": float(np.mean(aurocs)), "std": float(np.std(aurocs))}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-trials", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)

    # Load metadata
    meta_csv = NEUROVOZ_DIR / "metadata.csv"
    audio_dir = NEUROVOZ_DIR / "data" / "audios"
    if not meta_csv.exists():
        logger.error(f"metadata.csv not found at {meta_csv}")
        return

    # Load config from YAML files
    config = {}
    for name in ["model", "data", "train"]:
        cfg_path = PROJECT_ROOT / "configs" / f"{name}.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                config[name] = yaml.safe_load(f)

    # Load VoiceFM model
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    task_emb_key = "audio_encoder.task_embedding.weight"
    num_task_types = state[task_emb_key].shape[0] if task_emb_key in state else 100
    audio_encoder = build_audio_encoder(
        config=config["model"]["audio_encoder"],
        num_task_types=num_task_types,
    )
    ae_state = {k.replace("audio_encoder.", "", 1): v for k, v in state.items() if k.startswith("audio_encoder.")}
    audio_encoder.load_state_dict(ae_state)
    audio_encoder = audio_encoder.to(device)
    audio_encoder.eval()

    # Load frozen HuBERT baseline
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    hubert = hubert.to(device)
    hubert.eval()

    results = {"model": str(ckpt_path), "dataset": "NeuroVoz", "level": "participant"}

    # Evaluate per task category + all
    categories = [None, "vowel", "speech", "ddk"]
    cat_names = ["all", "vowel", "speech", "ddk"]

    for cat, cat_name in zip(categories, cat_names):
        logger.info(f"\n=== Task category: {cat_name} ===")
        ds = NeuroVozDataset(meta_csv, audio_dir, task_category=cat)
        if len(ds) < 20:
            logger.warning(f"Skipping {cat_name}: only {len(ds)} recordings")
            continue

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                           collate_fn=external_collate_fn, num_workers=2)

        # Extract recording-level embeddings
        vfm_emb, vfm_labels, vfm_pids = extract_voicefm_embeddings(audio_encoder, loader, device)
        hub_emb, hub_labels, hub_pids = extract_hubert_embeddings(hubert, loader, device)

        # Aggregate to participant level
        X_vfm, y_vfm, pids_vfm = aggregate_to_participants(vfm_emb, vfm_labels, vfm_pids)
        X_hub, y_hub, pids_hub = aggregate_to_participants(hub_emb, hub_labels, hub_pids)

        n_participants = len(pids_vfm)
        n_pd = int(np.sum(y_vfm == 1))
        n_hc = int(np.sum(y_vfm == 0))

        logger.info(f"  {len(ds)} recordings → {n_participants} participants ({n_pd} PD, {n_hc} HC)")

        # Linear probe (5-fold CV on participants)
        auroc_vfm, std_vfm = evaluate_probe(X_vfm, y_vfm)
        auroc_hub, std_hub = evaluate_probe(X_hub, y_hub)

        logger.info(f"  VoiceFM-256d: {auroc_vfm:.3f} ± {std_vfm:.3f}")
        logger.info(f"  HuBERT-768d:  {auroc_hub:.3f} ± {std_hub:.3f}")

        results[cat_name] = {
            "n_recordings": len(ds),
            "n_participants": n_participants,
            "n_pd": n_pd,
            "n_hc": n_hc,
            "voicefm_256d": {"auroc": auroc_vfm, "std": std_vfm},
            "hubert_768d": {"auroc": auroc_hub, "std": std_hub},
        }

        # Few-shot (only for 'all' and 'speech')
        if cat_name in ("all", "speech"):
            logger.info(f"  Few-shot evaluation ({cat_name})...")
            fs_vfm = few_shot_eval(X_vfm, y_vfm, n_trials=args.n_trials)
            fs_hub = few_shot_eval(X_hub, y_hub, n_trials=args.n_trials)
            results[f"{cat_name}_fewshot"] = {
                "voicefm_256d": fs_vfm,
                "hubert_768d": fs_hub,
            }
            for k in sorted(fs_vfm.keys()):
                logger.info(f"    k={k}: VFM-256={fs_vfm[k]['mean']:.3f} HuBERT={fs_hub[k]['mean']:.3f}")

    # Save results
    out_path = ckpt_path.parent / "neurovoz_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to {out_path}")

    # Summary table
    print("\n=== NeuroVoz Results Summary (Participant-Level) ===")
    print(f"{'Category':<15} {'Recs':>5} {'Ptcp':>5} {'VFM-256d':>10} {'HuBERT':>10}")
    print("-" * 55)
    for cat_name in cat_names:
        if cat_name not in results:
            continue
        r = results[cat_name]
        print(f"{cat_name:<15} {r['n_recordings']:>5} {r['n_participants']:>5} "
              f"{r['voicefm_256d']['auroc']:>10.3f} "
              f"{r['hubert_768d']['auroc']:>10.3f}")


if __name__ == "__main__":
    main()
