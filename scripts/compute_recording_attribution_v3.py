#!/usr/bin/env python3
"""Compute per-recording-type attribution analysis for VoiceFM.

Extracts all per-recording embeddings, then runs greedy forward selection
of recording types to answer: "Which recording types are most critical for
disease classification?"

Outputs a JSON file with:
  - per_task_aurocs: AUROC for each recording type (individual probes)
  - greedy_selection: cumulative AUROC curve from greedy forward selection
  - hubert_baseline: same analysis on frozen HuBERT embeddings

Usage:
    python scripts/compute_recording_attribution_v3.py \
        --checkpoint checkpoints_exp_h7_gsd_seed42/best_model.pt \
        --experiment exp_h7_gsd_seed42 \
        --baseline
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio_dataset import VoiceFMDataset, build_task_type_map, voicefm_collate_fn
from src.data.clinical_encoder import ClinicalFeatureProcessor
from src.data.sampler import create_participant_splits
from src.models import build_audio_encoder
from src.models.clinical_encoder import ClinicalEncoder
from src.models.voicefm import VoiceFM
from src.training.evaluate import extract_embeddings, build_label_dicts, extract_hubert_baseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CLF_LABELS = ["is_control", "cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]


# ---------------------------------------------------------------------------
# Config / model loading (same pattern as scripts/evaluate.py)
# ---------------------------------------------------------------------------

def deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(experiment: str | None = None) -> dict:
    import yaml
    config = {}
    for name in ["model", "data", "train"]:
        path = PROJECT_ROOT / "configs" / f"{name}.yaml"
        if path.exists():
            with open(path) as f:
                config[name] = yaml.safe_load(f)
    if experiment:
        exp_path = PROJECT_ROOT / "configs" / "experiments" / f"{experiment}.yaml"
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {exp_path}")
        with open(exp_path) as f:
            exp_overrides = yaml.safe_load(f)
        for section in ["model", "data", "train"]:
            if section in exp_overrides:
                config[section] = deep_merge(config[section], exp_overrides[section])
                del exp_overrides[section]
        if exp_overrides:
            config["train"] = deep_merge(config["train"], exp_overrides)
    return config


def build_model(model_cfg, num_task_types, device, use_gsd=False):
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
    return model.to(device)


# ---------------------------------------------------------------------------
# Attribution analysis functions
# ---------------------------------------------------------------------------

def aggregate_by_task_subset(
    embeddings: dict,
    task_subset: set[str],
) -> dict[str, np.ndarray]:
    """Mean-pool audio embeddings per participant, using only recordings from task_subset.

    Returns dict mapping participant_id -> mean embedding (256-dim).
    """
    audio = embeddings["audio_embeds"]
    pids = embeddings["participant_ids"]
    tasks = embeddings["task_names"]

    per_participant = defaultdict(list)
    for i, (pid, task) in enumerate(zip(pids, tasks)):
        if task in task_subset:
            per_participant[pid].append(audio[i])

    return {pid: np.mean(vecs, axis=0) for pid, vecs in per_participant.items() if vecs}


def probe_auroc_for_subset(
    train_embeds: dict,
    test_embeds: dict,
    train_labels: dict,
    test_labels: dict,
    task_subset: set[str],
) -> tuple[dict[str, float], float]:
    """Train linear probe on participant embeddings pooled from task_subset.

    Returns (aurocs_dict, coverage) where coverage = fraction of test participants included.
    """
    train_agg = aggregate_by_task_subset(train_embeds, task_subset)
    test_agg = aggregate_by_task_subset(test_embeds, task_subset)

    total_test = len(set(test_embeds["participant_ids"]))
    coverage = len(test_agg) / total_test if total_test > 0 else 0.0

    aurocs = {}
    for label_name in CLF_LABELS:
        X_train, y_train = _get_labeled_data(train_agg, train_labels, label_name)
        X_test, y_test = _get_labeled_data(test_agg, test_labels, label_name)

        if len(X_train) < 10 or len(X_test) < 5:
            continue
        if y_train.sum() < 3 or (len(y_train) - y_train.sum()) < 3:
            continue
        if y_test.sum() < 2 or (len(y_test) - y_test.sum()) < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train_s, y_train)
        y_prob = clf.predict_proba(X_test_s)[:, 1]
        aurocs[label_name] = float(roc_auc_score(y_test, y_prob))

    return aurocs, coverage


def _get_labeled_data(
    agg: dict[str, np.ndarray],
    labels: dict,
    label_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract aligned (X, y) arrays from aggregated embeddings and labels."""
    X, y = [], []
    for pid, embed in agg.items():
        if pid in labels and label_name in labels[pid]:
            val = labels[pid][label_name]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                X.append(embed)
                y.append(val)
    if not X:
        return np.array([]), np.array([])
    return np.array(X), np.array(y)


def greedy_task_selection(
    train_embeds: dict,
    test_embeds: dict,
    train_labels: dict,
    test_labels: dict,
    candidate_tasks: list[str],
) -> list[dict]:
    """Greedy forward selection of recording task types.

    At each step, try each remaining task type, pick the one that maximally
    improves mean AUROC across all 5 labels.

    Returns list of dicts with step, task_added, aurocs, mean_auroc, coverage.
    """
    selected = []
    remaining = set(candidate_tasks)
    history = []

    for step in range(len(candidate_tasks)):
        best_task = None
        best_score = -1
        best_aurocs = None
        best_coverage = 0.0

        for task in sorted(remaining):
            candidate_set = set(selected + [task])
            aurocs, coverage = probe_auroc_for_subset(
                train_embeds, test_embeds, train_labels, test_labels, candidate_set,
            )
            if not aurocs:
                continue
            mean_auroc = float(np.mean(list(aurocs.values())))
            if mean_auroc > best_score:
                best_task = task
                best_score = mean_auroc
                best_aurocs = aurocs
                best_coverage = coverage

        if best_task is None:
            break

        selected.append(best_task)
        remaining.remove(best_task)
        history.append({
            "step": step + 1,
            "task_added": best_task,
            "cumulative_tasks": list(selected),
            "aurocs": best_aurocs,
            "mean_auroc": best_score,
            "coverage": best_coverage,
        })
        logger.info(
            "Step %d: +%-35s  mean_auroc=%.3f  coverage=%.1f%%",
            step + 1, best_task, best_score, best_coverage * 100,
        )

        # Early stop if coverage is 100% and we've added at least 10 types
        if best_coverage >= 0.99 and step >= 9:
            # Check marginal improvement
            if len(history) >= 2 and (best_score - history[-2]["mean_auroc"]) < 0.001:
                logger.info("Stopping: marginal improvement < 0.001")
                break

    return history


def per_task_aurocs(
    train_embeds: dict,
    test_embeds: dict,
    train_labels: dict,
    test_labels: dict,
    min_test_participants: int = 20,
) -> dict:
    """Compute AUROC per recording type (individual probes, not cumulative)."""
    task_names = sorted(set(train_embeds.get("task_names", [])))
    test_task_counts = Counter(test_embeds.get("task_names", []))

    results = {}
    for task in task_names:
        # Count unique test participants for this task
        test_pids = set(
            pid for pid, t in zip(test_embeds["participant_ids"], test_embeds["task_names"])
            if t == task
        )
        if len(test_pids) < min_test_participants:
            continue

        aurocs, _ = probe_auroc_for_subset(
            train_embeds, test_embeds, train_labels, test_labels, {task},
        )
        if aurocs:
            aurocs["n_test"] = len(test_pids)
            aurocs["n_recordings_test"] = test_task_counts.get(task, 0)
            aurocs["mean_auroc"] = float(np.mean(list(
                v for k, v in aurocs.items() if k in CLF_LABELS
            )))
            results[task] = aurocs

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Recording attribution analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--baseline", action="store_true", help="Also analyze HuBERT baseline")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--min-participants", type=int, default=20)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    config = load_config(args.experiment)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load data
    data_dir = PROJECT_ROOT / "data" / "processed_v3"
    participants = pd.read_parquet(data_dir / "participants.parquet")
    recordings = pd.read_parquet(data_dir / "recordings.parquet")
    logger.info("Loaded %d participants, %d recordings", len(participants), len(recordings))

    audio_dir = PROJECT_ROOT / "data" / "audio"
    available_ids = {p.stem for p in audio_dir.glob("*.wav")}
    recordings = recordings[recordings["recording_id"].isin(available_ids)].reset_index(drop=True)
    participants = participants[participants.index.isin(recordings["record_id"].unique())]
    logger.info("After audio filter: %d participants, %d recordings", len(participants), len(recordings))

    use_gsd = config["data"].get("use_gsd", False)
    processor = ClinicalFeatureProcessor(use_gsd=use_gsd)
    feature_config = processor.get_feature_names()
    task_type_map = build_task_type_map(recordings)

    # Splits
    split_cfg = config["data"]["splits"]
    stratify_col = split_cfg.get("stratify_by")
    train_ids, val_ids, test_ids = create_participant_splits(
        participants,
        train_ratio=split_cfg["train"],
        val_ratio=split_cfg["val"],
        test_ratio=split_cfg["test"],
        seed=split_cfg["seed"],
        stratify_col=stratify_col,
    )
    logger.info("Splits: train=%d, val=%d, test=%d", len(train_ids), len(val_ids), len(test_ids))

    train_recordings = recordings[recordings["record_id"].isin(train_ids)]
    test_recordings = recordings[recordings["record_id"].isin(test_ids)]

    # Age stats
    train_participants = participants[participants.index.isin(train_ids)]
    train_ages = train_participants["age"].replace(-1, float("nan")).dropna()
    age_mean, age_std = float(train_ages.mean()), max(float(train_ages.std()), 1.0)

    # Datasets
    train_dataset = VoiceFMDataset(
        recording_manifest=train_recordings,
        participant_table=participants,
        audio_dir=audio_dir,
        task_type_map=task_type_map,
        feature_config=feature_config,
        age_mean=age_mean,
        age_std=age_std,
    )
    test_dataset = VoiceFMDataset(
        recording_manifest=test_recordings,
        participant_table=participants,
        audio_dir=audio_dir,
        task_type_map=task_type_map,
        feature_config=feature_config,
        age_mean=age_mean,
        age_std=age_std,
    )

    batch_size = config["train"]["training"]["batch_size"]
    num_workers = min(config["train"]["compute"]["num_workers"], 2)

    # All-recordings loaders (every recording, not 1 per participant)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        collate_fn=voicefm_collate_fn, num_workers=num_workers, shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        collate_fn=voicefm_collate_fn, num_workers=num_workers, shuffle=False,
    )

    # Build and load model
    model = build_model(config["model"], len(task_type_map) + 1, device, use_gsd=use_gsd)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_state = ckpt["model_state_dict"]
    model_state = model.state_dict()
    shape_mismatched = [
        k for k in ckpt_state
        if k in model_state and ckpt_state[k].shape != model_state[k].shape
    ]
    if shape_mismatched:
        logger.warning("Skipping %d keys with shape mismatch: %s",
                       len(shape_mismatched), shape_mismatched)
        for k in shape_mismatched:
            del ckpt_state[k]
    missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
    if missing:
        logger.warning("Missing keys (will use random init): %s", missing)
    if unexpected:
        logger.warning("Unexpected keys (ignored): %s", unexpected)
    logger.info("Loaded checkpoint: epoch %d", ckpt.get("epoch", -1))

    # Extract VoiceFM embeddings (all recordings)
    logger.info("Extracting VoiceFM embeddings (all train recordings)...")
    train_embeds = extract_embeddings(model, train_loader, device)
    logger.info("Extracting VoiceFM embeddings (all test recordings)...")
    test_embeds = extract_embeddings(model, test_loader, device)
    logger.info("Train: %d recordings, Test: %d recordings",
                len(train_embeds["recording_ids"]), len(test_embeds["recording_ids"]))

    # Build labels
    train_labels = build_label_dicts(participants, list(set(train_embeds["participant_ids"])))
    test_labels = build_label_dicts(participants, list(set(test_embeds["participant_ids"])))

    # --- Per-task AUROCs ---
    logger.info("Computing per-task AUROCs...")
    task_aurocs = per_task_aurocs(
        train_embeds, test_embeds, train_labels, test_labels,
        min_test_participants=args.min_participants,
    )
    logger.info("Evaluated %d recording types (min %d test participants)",
                len(task_aurocs), args.min_participants)

    # --- Greedy forward selection ---
    candidate_tasks = list(task_aurocs.keys())
    logger.info("Running greedy forward selection on %d candidate task types...", len(candidate_tasks))
    greedy_history = greedy_task_selection(
        train_embeds, test_embeds, train_labels, test_labels,
        candidate_tasks,
    )

    # --- All-types baseline (for comparison) ---
    all_aurocs, all_coverage = probe_auroc_for_subset(
        train_embeds, test_embeds, train_labels, test_labels,
        set(train_embeds["task_names"]),
    )

    result = {
        "seed": split_cfg["seed"],
        "model": "voicefm",
        "n_train_recordings": len(train_embeds["recording_ids"]),
        "n_test_recordings": len(test_embeds["recording_ids"]),
        "n_candidate_tasks": len(candidate_tasks),
        "per_task_aurocs": task_aurocs,
        "greedy_selection": greedy_history,
        "all_types_aurocs": all_aurocs,
        "all_types_coverage": all_coverage,
    }

    # Save VoiceFM results first (in case HuBERT baseline fails)
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = checkpoint_path.parent
        out_path = out_dir / "recording_attribution.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved VoiceFM attribution results to %s", out_path)

    # --- HuBERT baseline ---
    if args.baseline:
        logger.info("Extracting HuBERT baseline embeddings...")
        hub_train = extract_hubert_baseline(train_loader, device)
        hub_test = extract_hubert_baseline(test_loader, device)

        logger.info("Computing HuBERT per-task AUROCs...")
        hub_task_aurocs = per_task_aurocs(
            hub_train, hub_test, train_labels, test_labels,
            min_test_participants=args.min_participants,
        )

        hub_candidates = list(hub_task_aurocs.keys())
        logger.info("Running HuBERT greedy forward selection on %d types...", len(hub_candidates))
        hub_greedy = greedy_task_selection(
            hub_train, hub_test, train_labels, test_labels,
            hub_candidates,
        )

        hub_all_aurocs, hub_all_coverage = probe_auroc_for_subset(
            hub_train, hub_test, train_labels, test_labels,
            set(hub_train["task_names"]),
        )

        result["hubert_baseline"] = {
            "per_task_aurocs": hub_task_aurocs,
            "greedy_selection": hub_greedy,
            "all_types_aurocs": hub_all_aurocs,
            "all_types_coverage": hub_all_coverage,
        }

    # Re-save with HuBERT baseline included
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved final attribution results to %s", out_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Recording Attribution Summary (seed {split_cfg['seed']})")
    print(f"{'='*60}")
    print(f"  Recording types evaluated: {len(task_aurocs)}")
    print(f"  Greedy selection steps: {len(greedy_history)}")
    if greedy_history:
        print(f"\n  Top 5 recording types (greedy order):")
        for h in greedy_history[:5]:
            print(f"    {h['step']}. {h['task_added']:<35s} mean_auroc={h['mean_auroc']:.3f} "
                  f"coverage={h['coverage']:.0%}")
        print(f"\n  All-types AUROC: {all_aurocs}")


if __name__ == "__main__":
    main()
