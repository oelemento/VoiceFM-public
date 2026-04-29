#!/usr/bin/env python3
"""Evaluate VoiceFM embeddings: retrieval, linear probes, UMAP, HuBERT baseline.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
    python scripts/evaluate.py --checkpoint checkpoints_exp_d_hard_negatives/best_model.pt --baseline
"""

import argparse
import json
import sys
import logging
from pathlib import Path

import torch
import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio_dataset import VoiceFMDataset, build_task_type_map, voicefm_collate_fn
from src.data.clinical_encoder import ClinicalFeatureProcessor
from src.data.sampler import (
    ParticipantBatchSampler,
    build_participant_strata,
    create_participant_splits,
)
from src.models import build_audio_encoder
from src.models.clinical_encoder import ClinicalEncoder
from src.models.voicefm import VoiceFM
from src.training.evaluate import (
    extract_embeddings,
    retrieval_evaluation,
    linear_probe_evaluation,
    task_stratified_probe_evaluation,
    build_label_dicts,
    plot_umap,
    plot_comparison_figures,
    plot_task_stratified_heatmap,
    extract_hubert_baseline,
    extract_hear_baseline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(experiment: str | None = None) -> dict:
    """Load and merge config files, with optional experiment overrides."""
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
        logger.info("Loading experiment: %s", experiment)
        logger.info("Overrides: %s", exp_overrides)
        for section in ["model", "data", "train"]:
            if section in exp_overrides:
                config[section] = deep_merge(config[section], exp_overrides[section])
                del exp_overrides[section]
        if exp_overrides:
            config["train"] = deep_merge(config["train"], exp_overrides)

    return config


def build_model(model_cfg: dict, num_task_types: int, device: torch.device,
                 use_gsd: bool = False, exclude_features: list[str] | None = None) -> VoiceFM:
    """Build VoiceFM model from config (no augmentation/checkpointing for eval)."""
    feature_config = ClinicalFeatureProcessor(use_gsd=use_gsd).get_feature_names()
    if exclude_features:
        feature_config["binary"] = [f for f in feature_config["binary"] if f not in exclude_features]
        feature_config["continuous"] = [f for f in feature_config["continuous"] if f not in exclude_features]
        feature_config["categorical"] = {k: v for k, v in feature_config["categorical"].items() if k not in exclude_features}

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


def print_results(title: str, metrics: dict):
    """Print metrics in a clean table."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key:<45s} {value:.4f}")
        else:
            print(f"  {key:<45s} {value}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VoiceFM")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (e.g. checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Also evaluate raw HuBERT embeddings for comparison",
    )
    parser.add_argument(
        "--hear-baseline", action="store_true",
        help="Also evaluate raw HeAR embeddings for comparison",
    )
    parser.add_argument(
        "--no-umap", action="store_true",
        help="Skip UMAP visualization (faster)",
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Experiment config name (e.g. exp_i_hear). "
             "Loaded from configs/experiments/<name>.yaml",
    )
    parser.add_argument(
        "--output-name", type=str, default=None,
        help="Override output JSON filename (e.g. crosseval_gsd_results.json). "
             "Saved in the checkpoint's parent directory.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data directory (default: data/processed). "
             "Used to route v3 retraining to data/processed_v3 without touching v2.3.0 artifacts.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        # Try relative to PROJECT_ROOT
        checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    config = load_config(args.experiment)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # Load data
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "processed"
    participants = pd.read_parquet(data_dir / "participants.parquet")
    recordings = pd.read_parquet(data_dir / "recordings.parquet")
    logger.info("Loaded %d participants, %d recordings", len(participants), len(recordings))

    # Filter to available audio
    audio_dir = PROJECT_ROOT / "data" / "audio"
    available_ids = {p.stem for p in audio_dir.glob("*.wav")}
    recordings = recordings[recordings["recording_id"].isin(available_ids)].reset_index(drop=True)
    participants = participants[participants.index.isin(recordings["record_id"].unique())]
    logger.info("After audio filter: %d participants, %d recordings", len(participants), len(recordings))

    # Feature config and task types
    use_gsd = config["data"].get("use_gsd", False)
    processor = ClinicalFeatureProcessor(use_gsd=use_gsd)
    feature_config = processor.get_feature_names()
    exclude = config["data"].get("exclude_features", [])
    if exclude:
        feature_config["binary"] = [f for f in feature_config["binary"] if f not in exclude]
        feature_config["continuous"] = [f for f in feature_config["continuous"] if f not in exclude]
        feature_config["categorical"] = {k: v for k, v in feature_config["categorical"].items() if k not in exclude}
        logger.info(f"Excluded {len(exclude)} features")
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
    logger.info(
        "Splits: train=%d, val=%d, test=%d (stratify_by=%s)",
        len(train_ids), len(val_ids), len(test_ids), stratify_col,
    )

    train_recordings = recordings[recordings["record_id"].isin(train_ids)]
    test_recordings = recordings[recordings["record_id"].isin(test_ids)]
    participant_strata = build_participant_strata(participants, stratify_col)
    use_category_stratify = config["train"]["training"].get("category_stratify", True)
    train_categories = (
        participant_strata.loc[train_ids] if use_category_stratify and participant_strata is not None else None
    )
    test_categories = (
        participant_strata.loc[test_ids] if use_category_stratify and participant_strata is not None else None
    )

    # Age stats from train
    train_participants = participants[participants.index.isin(train_ids)]
    train_ages = train_participants["age"].replace(-1, float("nan")).dropna()
    age_mean = float(train_ages.mean())
    age_std = float(train_ages.std()) if train_ages.std() > 0 else 1.0

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

    # Samplers (no drop_last for eval) — 1 recording per participant
    batch_size = config["train"]["training"]["batch_size"]
    train_sampler = ParticipantBatchSampler(
        train_recordings,
        participant_categories=train_categories,
        batch_size=batch_size,
        task_stratify=False,
        drop_last=False,
    )
    test_sampler = ParticipantBatchSampler(
        test_recordings,
        participant_categories=test_categories,
        batch_size=batch_size,
        task_stratify=False,
        drop_last=False,
    )

    num_workers = min(config["train"]["compute"]["num_workers"], 2)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_sampler,
        collate_fn=voicefm_collate_fn, num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_sampler=test_sampler,
        collate_fn=voicefm_collate_fn, num_workers=num_workers,
    )

    # All-recordings loaders for task-stratified eval (every recording, not 1 per participant)
    train_loader_all = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        collate_fn=voicefm_collate_fn, num_workers=num_workers, shuffle=False,
    )
    test_loader_all = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        collate_fn=voicefm_collate_fn, num_workers=num_workers, shuffle=False,
    )

    # Build and load model
    model = build_model(config["model"], len(task_type_map) + 1, device,
                        use_gsd=use_gsd, exclude_features=exclude)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Filter out keys with shape mismatches (e.g. cross-eval: SR model → GSD config)
    ckpt_state = ckpt["model_state_dict"]
    model_state = model.state_dict()
    shape_mismatched = [
        k for k in ckpt_state
        if k in model_state and ckpt_state[k].shape != model_state[k].shape
    ]
    if shape_mismatched:
        logger.warning("Skipping %d keys with shape mismatch (cross-eval): %s",
                       len(shape_mismatched), shape_mismatched)
        for k in shape_mismatched:
            del ckpt_state[k]
    missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
    if missing:
        logger.warning("Missing keys (will use random init): %s", missing)
    if unexpected:
        logger.warning("Unexpected keys (ignored): %s", unexpected)
    logger.info("Loaded checkpoint from epoch %d (val_loss=%.4f)",
                ckpt.get("epoch", -1), ckpt.get("val_loss", float("nan")))

    # --- Extract VoiceFM embeddings ---
    logger.info("Extracting VoiceFM embeddings (train)...")
    train_embeds = extract_embeddings(model, train_loader, device)
    logger.info("Extracting VoiceFM embeddings (test)...")
    test_embeds = extract_embeddings(model, test_loader, device)

    # --- Retrieval ---
    logger.info("Running retrieval evaluation...")
    retrieval_metrics = retrieval_evaluation(test_embeds)
    print_results("Retrieval Metrics (Test Set)", retrieval_metrics)

    # --- Build labels ---
    train_labels = build_label_dicts(participants, list(set(train_embeds["participant_ids"])))
    test_labels = build_label_dicts(participants, list(set(test_embeds["participant_ids"])))

    # --- Linear probes ---
    logger.info("Running linear probe evaluation...")
    probe_metrics, voicefm_curves = linear_probe_evaluation(
        train_embeds, test_embeds, train_labels, test_labels, return_curves=True,
    )
    print_results("Linear Probe Metrics (VoiceFM)", probe_metrics)

    all_metrics = {**retrieval_metrics, **{f"voicefm/{k}": v for k, v in probe_metrics.items()}}

    # --- Task-stratified linear probes (uses ALL recordings, not 1 per participant) ---
    logger.info("Extracting all-recordings embeddings for task-stratified eval...")
    train_embeds_all = extract_embeddings(model, train_loader_all, device)
    test_embeds_all = extract_embeddings(model, test_loader_all, device)
    logger.info("All-recordings: train=%d, test=%d",
                len(train_embeds_all["recording_ids"]), len(test_embeds_all["recording_ids"]))

    logger.info("Running task-stratified probe evaluation...")
    try:
        stratified_metrics = task_stratified_probe_evaluation(
            train_embeds_all, test_embeds_all, train_labels, test_labels,
        )
        # Print summary
        summary = stratified_metrics.pop("task_stratified/_summary", {})
        if summary:
            print(f"\n{'=' * 80}")
            print("  Task-Stratified AUROC (recording type → classification task)")
            print(f"{'=' * 80}")
            clf_labels = ["is_control", "cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]
            header = f"  {'Task Type':<35s}" + "".join(f"{l:>14s}" for l in clf_labels) + f"{'n_test':>8s}"
            print(header)
            print(f"  {'-' * (len(header) - 2)}")
            for task_name in sorted(summary.keys()):
                vals = summary[task_name]
                row = f"  {task_name[:35]:<35s}"
                for l in clf_labels:
                    if l in vals:
                        row += f"{vals[l]:14.3f}"
                    else:
                        row += f"{'—':>14s}"
                row += f"{vals.get('n_test', '?'):>8}"
                print(row)
            print(f"{'=' * 80}")

            # Save heatmap
            stratified_metrics["task_stratified/_summary"] = summary
            heatmap_path = checkpoint_path.parent / f"task_stratified_{checkpoint_path.stem}.png"
            plot_task_stratified_heatmap(stratified_metrics, heatmap_path)

        all_metrics.update({k: v for k, v in stratified_metrics.items() if k != "task_stratified/_summary"})
    except Exception as e:
        logger.warning("Task-stratified evaluation failed: %s", e, exc_info=True)

    # --- UMAP ---
    if not args.no_umap:
        try:
            logger.info("Generating UMAP visualization...")
            umap_path = checkpoint_path.parent / f"umap_{checkpoint_path.stem}.png"
            plot_umap(test_embeds, participants, umap_path)
        except Exception as e:
            logger.warning("UMAP generation failed: %s", e)

    # --- HuBERT baseline ---
    if args.baseline:
        logger.info("Extracting HuBERT baseline embeddings (train)...")
        train_baseline = extract_hubert_baseline(train_loader, device)
        logger.info("Extracting HuBERT baseline embeddings (test)...")
        test_baseline = extract_hubert_baseline(test_loader, device)

        # Clinical embeds not available for baseline — only run probes on audio
        logger.info("Running linear probes on HuBERT baseline...")
        baseline_probe_metrics, hubert_curves = linear_probe_evaluation(
            train_baseline, test_baseline, train_labels, test_labels, return_curves=True,
        )
        print_results("Linear Probe Metrics (HuBERT Baseline)", baseline_probe_metrics)

        for k, v in baseline_probe_metrics.items():
            all_metrics[f"hubert_baseline/{k}"] = v

        # Print comparison
        print(f"\n{'=' * 60}")
        print("  COMPARISON: VoiceFM vs HuBERT Baseline")
        print(f"{'=' * 60}")
        print(f"  {'Metric':<40s} {'VoiceFM':>8s} {'HuBERT':>8s} {'Delta':>8s}")
        print(f"  {'-' * 56}")
        for key in sorted(probe_metrics.keys()):
            vm = probe_metrics.get(key, float("nan"))
            hm = baseline_probe_metrics.get(key, float("nan"))
            delta = vm - hm
            print(f"  {key:<40s} {vm:8.4f} {hm:8.4f} {delta:+8.4f}")
        print(f"{'=' * 60}")

        # --- Comparison figures ---
        try:
            logger.info("Generating comparison figures...")
            figure_paths = plot_comparison_figures(
                voicefm_metrics=probe_metrics,
                hubert_metrics=baseline_probe_metrics,
                voicefm_curves=voicefm_curves,
                hubert_curves=hubert_curves,
                save_dir=checkpoint_path.parent,
            )
            logger.info("Saved %d comparison figures to %s", len(figure_paths), checkpoint_path.parent)
        except Exception as e:
            logger.warning("Comparison figure generation failed: %s", e, exc_info=True)

    # --- HeAR baseline ---
    if args.hear_baseline:
        logger.info("Extracting HeAR baseline embeddings (train)...")
        train_hear = extract_hear_baseline(train_loader, device)
        logger.info("Extracting HeAR baseline embeddings (test)...")
        test_hear = extract_hear_baseline(test_loader, device)

        logger.info("Running linear probes on HeAR baseline...")
        hear_probe_metrics, hear_curves = linear_probe_evaluation(
            train_hear, test_hear, train_labels, test_labels, return_curves=True,
        )
        print_results("Linear Probe Metrics (HeAR Baseline)", hear_probe_metrics)

        for k, v in hear_probe_metrics.items():
            all_metrics[f"hear_baseline/{k}"] = v

        # Print comparison
        print(f"\n{'=' * 60}")
        print("  COMPARISON: VoiceFM vs HeAR Baseline")
        print(f"{'=' * 60}")
        print(f"  {'Metric':<40s} {'VoiceFM':>8s} {'HeAR':>8s} {'Delta':>8s}")
        print(f"  {'-' * 56}")
        for key in sorted(probe_metrics.keys()):
            vm = probe_metrics.get(key, float("nan"))
            hm = hear_probe_metrics.get(key, float("nan"))
            delta = vm - hm
            print(f"  {key:<40s} {vm:8.4f} {hm:8.4f} {delta:+8.4f}")
        print(f"{'=' * 60}")

    # Save results
    if args.output_name:
        results_path = checkpoint_path.parent / args.output_name
    else:
        results_path = checkpoint_path.parent / f"eval_results_{checkpoint_path.stem}.json"
    with open(results_path, "w") as f:
        json.dump({k: float(v) if isinstance(v, (float, int)) else v
                   for k, v in all_metrics.items()}, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
