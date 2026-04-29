#!/usr/bin/env python3
"""H2: Few-shot evaluation on external datasets.

Tests whether VoiceFM can detect conditions from just a few labeled examples,
demonstrating foundation model utility.

For each external dataset and each k in [1, 2, 5, 10, 20]:
  - Sample k positive + k negative examples for training
  - Evaluate on remaining samples
  - Repeat 100 times
  - Report mean ± std AUROC

Compares VoiceFM (GSD) vs frozen HuBERT baseline.

Usage:
    python scripts/evaluate_fewshot_v3.py \
        --checkpoint checkpoints_exp_d_gsd_v3_seed42/best_model.pt \
        --experiment exp_d_gsd_v3_seed42
"""

import argparse
import json
import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import HubertModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.external_datasets import (
    CoswaraDataset, SVDDataset, VOICEDDataset,
    FigsharePDDataset, MDVRKCLDataset, MPowerDataset,
    CombinedExternalDataset, external_collate_fn,
)
from src.models import build_audio_encoder
from src.models.voicefm import VoiceFM
from src.models.clinical_encoder import ClinicalEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base without clobbering nested keys."""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(experiment=None):
    config = {}
    for name in ["model", "data", "train"]:
        path = PROJECT_ROOT / "configs" / f"{name}.yaml"
        if path.exists():
            with open(path) as f:
                config[name] = yaml.safe_load(f)
    if experiment:
        exp_path = PROJECT_ROOT / "configs" / "experiments" / f"{experiment}.yaml"
        if exp_path.exists():
            with open(exp_path) as f:
                exp_overrides = yaml.safe_load(f) or {}
            for section in ["model", "data", "train", "loss"]:
                if section in exp_overrides:
                    config[section] = deep_merge(
                        config.get(section, {}), exp_overrides[section]
                    )
    return config


def build_voicefm_model(config, checkpoint_path, device, use_gsd=False):
    from src.data.clinical_encoder import ClinicalFeatureProcessor
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    task_emb_key = "audio_encoder.task_embedding.weight"
    num_task_types = state_dict[task_emb_key].shape[0] if task_emb_key in state_dict else 100

    model_cfg = config["model"]
    feature_config = ClinicalFeatureProcessor(use_gsd=use_gsd).get_feature_names()

    audio_encoder = build_audio_encoder(
        config=model_cfg["audio_encoder"],
        num_task_types=num_task_types,
        spec_augment=False,
        gradient_checkpointing=False,
    )
    clinical_encoder = ClinicalEncoder(
        feature_config=feature_config,
        num_layers=model_cfg["clinical_encoder"]["num_layers"],
        num_heads=model_cfg["clinical_encoder"]["num_heads"],
        hidden_dim=model_cfg["clinical_encoder"]["hidden_dim"],
        dropout=model_cfg["clinical_encoder"]["dropout"],
        projection_dim=model_cfg["clinical_encoder"]["projection_dim"],
    )
    model = VoiceFM(
        audio_encoder=audio_encoder,
        clinical_encoder=clinical_encoder,
        temperature_init=model_cfg["contrastive"]["temperature_init"],
        learn_temperature=model_cfg["contrastive"]["learn_temperature"],
    )

    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    model = model.to(device).eval()
    logger.info("Loaded VoiceFM: epoch=%d", ckpt.get("epoch", -1))
    return model


@torch.no_grad()
def extract_voicefm_embeddings(model, dataloader, device):
    model.eval()
    all_embeds, all_labels, all_ds_ids = [], [], []
    for batch in dataloader:
        audio = batch["audio_values"].to(device)
        mask = batch["attention_mask"].to(device)
        task_ids = torch.zeros(audio.shape[0], dtype=torch.long, device=device)
        embeds = model.audio_encoder(
            audio_input_values=audio, attention_mask=mask, task_type_ids=task_ids,
        )
        all_embeds.append(embeds.cpu().numpy())
        all_labels.append(batch["labels"]["disease"].numpy())
        all_ds_ids.extend(batch["dataset_ids"].tolist())
    return np.concatenate(all_embeds), np.concatenate(all_labels), all_ds_ids


@torch.no_grad()
def extract_hubert_embeddings(dataloader, device):
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    all_embeds, all_labels, all_ds_ids = [], [], []
    for batch in dataloader:
        audio = batch["audio_values"].to(device)
        mask = batch["attention_mask"].to(device)
        output = hubert(input_values=audio, attention_mask=mask, return_dict=True)
        hidden = output.last_hidden_state
        frame_mask = hubert._get_feature_vector_attention_mask(hidden.shape[1], mask)
        frame_mask_f = frame_mask.unsqueeze(-1).float()
        pooled = (hidden * frame_mask_f).sum(dim=1) / frame_mask_f.sum(dim=1).clamp(min=1)
        pooled = F.normalize(pooled, p=2, dim=-1)
        all_embeds.append(pooled.cpu().numpy())
        all_labels.append(batch["labels"]["disease"].numpy())
        all_ds_ids.extend(batch["dataset_ids"].tolist())
    return np.concatenate(all_embeds), np.concatenate(all_labels), all_ds_ids


def few_shot_probe(X, y, k, n_trials=100, seed=42):
    """Run n_trials of k-shot classification.

    Samples k positive + k negative for training, evaluates on remaining.
    """
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    # Need at least k+2 per class (k for train, 2+ for test)
    if len(pos_idx) < k + 2 or len(neg_idx) < k + 2:
        return {"mean_auroc": float("nan"), "std_auroc": 0, "n_trials": 0,
                "k": k, "n_pos": len(pos_idx), "n_neg": len(neg_idx)}

    aurocs = []
    for _ in range(n_trials):
        train_pos = rng.choice(pos_idx, k, replace=False)
        train_neg = rng.choice(neg_idx, k, replace=False)
        train_idx = np.concatenate([train_pos, train_neg])

        test_mask = np.ones(len(y), dtype=bool)
        test_mask[train_idx] = False
        test_idx = np.where(test_mask)[0]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        if len(set(y_test)) < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_train_s, y_train)

        y_prob = clf.predict_proba(X_test_s)[:, 1]
        try:
            aurocs.append(roc_auc_score(y_test, y_prob))
        except ValueError:
            continue

    if not aurocs:
        return {"mean_auroc": float("nan"), "std_auroc": 0, "n_trials": 0,
                "k": k, "n_pos": len(pos_idx), "n_neg": len(neg_idx)}

    return {
        "mean_auroc": float(np.mean(aurocs)),
        "std_auroc": float(np.std(aurocs)),
        "n_trials": len(aurocs),
        "k": k,
        "n_pos": len(pos_idx),
        "n_neg": len(neg_idx),
    }


def load_external_datasets():
    datasets = []
    for name, cls, subdir in [
        ("Coswara", CoswaraDataset, "coswara"),
        ("SVD", SVDDataset, "svd"),
        ("VOICED", VOICEDDataset, "voiced"),
        ("MDVR-KCL", MDVRKCLDataset, "mdvr_kcl"),
    ]:
        d = EXTERNAL_DIR / subdir
        meta = d / "metadata.csv"
        if not meta.exists() and subdir == "svd":
            meta = d / "data" / "metadata" / "metadata.csv"
        if meta.exists():
            ds = cls(meta, d / "audio")
            if len(ds) > 0:
                datasets.append((name, ds))
    return datasets


def main():
    parser = argparse.ArgumentParser(description="Few-shot evaluation on external datasets")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-trials", type=int, default=100)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / args.checkpoint
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_values = [1, 2, 5, 10, 20]

    # Load datasets
    ext_datasets = load_external_datasets()
    if not ext_datasets:
        logger.error("No external datasets found in %s", EXTERNAL_DIR)
        sys.exit(1)

    logger.info("Found %d external datasets", len(ext_datasets))
    for name, ds in ext_datasets:
        labels = ds.metadata[ds.DISEASE_COLUMN].astype(int)
        logger.info("  %s: %d samples (%d pos, %d neg)", name, len(ds), labels.sum(), len(ds) - labels.sum())

    # Combined loader
    combined = CombinedExternalDataset([ds for _, ds in ext_datasets])
    loader = DataLoader(
        combined, batch_size=args.batch_size,
        collate_fn=external_collate_fn, num_workers=2, shuffle=False,
    )

    # Extract embeddings
    logger.info("=== Extracting VoiceFM embeddings ===")
    config = load_config(args.experiment)
    use_gsd = config.get("data", {}).get("use_gsd", False)
    model = build_voicefm_model(config, ckpt_path, device, use_gsd=use_gsd)
    vfm_embeds, vfm_labels, vfm_ds_ids = extract_voicefm_embeddings(model, loader, device)
    del model
    torch.cuda.empty_cache()

    logger.info("=== Extracting HuBERT embeddings ===")
    hub_embeds, hub_labels, hub_ds_ids = extract_hubert_embeddings(loader, device)

    # Few-shot probing
    all_results = {}

    print(f"\n{'=' * 100}")
    print(f"  Few-Shot Evaluation: k-shot probing on external datasets ({args.n_trials} trials per k)")
    print(f"{'=' * 100}")

    for ds_name, ds in ext_datasets:
        ds_id = ds.DATASET_ID
        mask = np.array([d == ds_id for d in vfm_ds_ids])
        X_vfm = vfm_embeds[mask]
        X_hub = hub_embeds[mask]
        y = vfm_labels[mask]

        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        print(f"\n  {ds_name} (N={len(y)}, pos={n_pos}, neg={n_neg})")
        print(f"  {'k':>5s}  {'VoiceFM AUROC':>20s}  {'HuBERT AUROC':>20s}  {'Delta':>10s}")
        print(f"  {'-' * 60}")

        ds_results = {}
        for k in k_values:
            vfm_res = few_shot_probe(X_vfm, y, k, n_trials=args.n_trials)
            hub_res = few_shot_probe(X_hub, y, k, n_trials=args.n_trials)

            vfm_str = f"{vfm_res['mean_auroc']:.3f}±{vfm_res['std_auroc']:.3f}" if not np.isnan(vfm_res['mean_auroc']) else "N/A"
            hub_str = f"{hub_res['mean_auroc']:.3f}±{hub_res['std_auroc']:.3f}" if not np.isnan(hub_res['mean_auroc']) else "N/A"
            delta = vfm_res['mean_auroc'] - hub_res['mean_auroc'] if not np.isnan(vfm_res['mean_auroc']) and not np.isnan(hub_res['mean_auroc']) else float("nan")
            delta_str = f"{delta:+.3f}" if not np.isnan(delta) else "N/A"

            print(f"  {k:>5d}  {vfm_str:>20s}  {hub_str:>20s}  {delta_str:>10s}")

            ds_results[f"k={k}"] = {
                "voicefm": vfm_res,
                "hubert": hub_res,
                "delta": float(delta) if not np.isnan(delta) else None,
            }

        all_results[ds_name] = ds_results

    # Summary: average across datasets for each k
    print(f"\n  {'MEAN across datasets':}")
    print(f"  {'k':>5s}  {'VoiceFM AUROC':>20s}  {'HuBERT AUROC':>20s}  {'Delta':>10s}  {'Wins':>5s}")
    print(f"  {'-' * 65}")
    for k in k_values:
        vfm_aurocs, hub_aurocs = [], []
        wins = 0
        for ds_name in all_results:
            res = all_results[ds_name].get(f"k={k}")
            if res and res["delta"] is not None:
                vfm_aurocs.append(res["voicefm"]["mean_auroc"])
                hub_aurocs.append(res["hubert"]["mean_auroc"])
                if res["delta"] > 0:
                    wins += 1
        if vfm_aurocs:
            print(f"  {k:>5d}  {np.mean(vfm_aurocs):>14.3f}       {np.mean(hub_aurocs):>14.3f}       {np.mean(vfm_aurocs) - np.mean(hub_aurocs):>+10.3f}  {wins}/{len(vfm_aurocs)}")

    print(f"{'=' * 100}")

    # Save results
    save_path = ckpt_path.parent / f"fewshot_results_{ckpt_path.stem}.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", save_path)


if __name__ == "__main__":
    main()
