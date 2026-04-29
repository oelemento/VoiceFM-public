"""Evaluation suite for VoiceFM embeddings."""

import logging
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import HubertModel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Extract audio and clinical embeddings from the model.

    Returns:
        Dict with keys: audio_embeds, clinical_embeds, participant_ids,
        recording_ids, task_names
    """
    model.eval()
    results = {
        "audio_embeds": [],
        "clinical_embeds": [],
        "participant_ids": [],
        "recording_ids": [],
        "task_names": [],
    }

    for batch in dataloader:
        audio = batch["audio_input_values"].to(device)
        mask = batch["attention_mask"].to(device)
        task_ids = batch["task_type_id"].to(device)

        clinical = {}
        for k, v in batch["clinical_features"].items():
            if isinstance(v, torch.Tensor):
                clinical[k] = v.to(device)

        output = model(audio, mask, task_ids, clinical)

        results["audio_embeds"].append(output["audio_embeds"].cpu())
        results["clinical_embeds"].append(output["clinical_embeds"].cpu())
        results["participant_ids"].extend(batch["participant_id"])
        results["recording_ids"].extend(batch["recording_id"])
        results["task_names"].extend(batch["task_name"])

    results["audio_embeds"] = torch.cat(results["audio_embeds"]).numpy()
    results["clinical_embeds"] = torch.cat(results["clinical_embeds"]).numpy()

    return results


def retrieval_evaluation(embeddings: dict) -> dict:
    """Cross-modal retrieval evaluation.

    Computes Recall@K for audio->clinical and clinical->audio.
    Aggregates at participant level (average embeddings per participant).
    """
    audio = embeddings["audio_embeds"]
    clinical = embeddings["clinical_embeds"]
    pids = embeddings["participant_ids"]

    # Aggregate per participant
    unique_pids = sorted(set(pids))
    pid_audio = np.zeros((len(unique_pids), audio.shape[1]))
    pid_clinical = np.zeros((len(unique_pids), clinical.shape[1]))

    for i, pid in enumerate(unique_pids):
        mask = [j for j, p in enumerate(pids) if p == pid]
        pid_audio[i] = audio[mask].mean(axis=0)
        pid_clinical[i] = clinical[mask].mean(axis=0)

    # Normalize
    pid_audio = pid_audio / np.linalg.norm(pid_audio, axis=1, keepdims=True)
    pid_clinical = pid_clinical / np.linalg.norm(pid_clinical, axis=1, keepdims=True)

    sim = pid_audio @ pid_clinical.T
    n = len(unique_pids)

    metrics = {}
    for k in [1, 5, 10]:
        # Audio -> Clinical
        topk = np.argsort(-sim, axis=1)[:, :k]
        correct = np.array([i in topk[i] for i in range(n)])
        metrics[f"participant_recall@{k}_a2c"] = correct.mean()

        # Clinical -> Audio
        topk = np.argsort(-sim.T, axis=1)[:, :k]
        correct = np.array([i in topk[i] for i in range(n)])
        metrics[f"participant_recall@{k}_c2a"] = correct.mean()

    return metrics


def linear_probe_evaluation(
    train_embeddings: dict,
    test_embeddings: dict,
    train_labels: dict,
    test_labels: dict,
    return_curves: bool = False,
) -> dict | tuple[dict, dict]:
    """Linear probe evaluation of embedding quality.

    Trains simple linear classifiers/regressors on frozen embeddings.

    Args:
        train_embeddings, test_embeddings: Output of extract_embeddings
        train_labels, test_labels: Dicts mapping participant_id -> label dict
            Each label dict has keys like 'is_control', 'cat_voice', 'age', etc.
        return_curves: If True, return (metrics, curves) where curves contains
            y_true/y_prob/y_pred for each task (needed for ROC plots).
    """
    # Aggregate embeddings per participant
    train_agg = _aggregate_per_participant(train_embeddings)
    test_agg = _aggregate_per_participant(test_embeddings)

    metrics = {}
    curves = {}

    # Classification probes — core categories + any GSD diagnosis flags
    core_labels = ["is_control", "cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]
    # Discover additional binary labels (e.g., gsd_parkinsons, gsd_mtd, etc.)
    sample_labels = next(iter(train_labels.values()), {})
    extra_labels = [k for k in sample_labels
                    if k not in core_labels and k not in ("age", "phq9_score", "gad7_score")
                    and isinstance(sample_labels[k], (int, float)) and sample_labels[k] in (0, 1)]
    for label_name in core_labels + sorted(extra_labels):
        X_train, y_train = _get_labeled_data(train_agg, train_labels, label_name)
        X_test, y_test = _get_labeled_data(test_agg, test_labels, label_name)

        if len(X_train) == 0 or len(X_test) == 0:
            continue
        # Skip labels with too few positives/negatives for reliable probing
        train_pos = int(y_train.sum())
        test_pos = int(y_test.sum())
        if train_pos < 5 or (len(y_train) - train_pos) < 5:
            continue
        if test_pos < 2 or (len(y_test) - test_pos) < 2:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] if len(set(y_test)) == 2 else None

        metrics[f"probe/{label_name}/accuracy"] = accuracy_score(y_test, y_pred)
        metrics[f"probe/{label_name}/f1"] = f1_score(y_test, y_pred, average="macro")
        if y_prob is not None and len(set(y_test)) == 2:
            metrics[f"probe/{label_name}/auroc"] = roc_auc_score(y_test, y_prob)

        if return_curves:
            curves[label_name] = {
                "y_true": y_test.tolist(),
                "y_pred": y_pred.tolist(),
                "y_prob": y_prob.tolist() if y_prob is not None else None,
                "n_train": len(y_train),
                "n_test": len(y_test),
            }

    # Regression probes
    for label_name in ["age", "phq9_score", "gad7_score"]:
        X_train, y_train = _get_labeled_data(train_agg, train_labels, label_name)
        X_test, y_test = _get_labeled_data(test_agg, test_labels, label_name)

        # Filter out missing
        valid_train = ~np.isnan(y_train)
        valid_test = ~np.isnan(y_test)
        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_test, y_test = X_test[valid_test], y_test[valid_test]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        reg = Ridge(alpha=1.0)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        metrics[f"probe/{label_name}/rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics[f"probe/{label_name}/r2"] = reg.score(X_test, y_test)

        if return_curves:
            curves[label_name] = {
                "y_true": y_test.tolist(),
                "y_pred": y_pred.tolist(),
                "n_train": len(y_train),
                "n_test": len(y_test),
            }

    if return_curves:
        return metrics, curves
    return metrics


def build_label_dicts(
    participants: pd.DataFrame,
    participant_ids: list[str],
) -> dict:
    """Build label dict for linear probes from participants table.

    Args:
        participants: DataFrame indexed by record_id with columns like
            is_control_participant, cat_voice, cat_neuro, etc.
        participant_ids: List of participant IDs to include.

    Returns:
        Dict mapping participant_id -> dict of label_name -> value.
        Missing questionnaire scores (sentinel -1) mapped to None.
    """
    # Individual GSD diagnosis flags to probe (if present in participants)
    gsd_flags = [
        "gsd_glottic_insufficiency", "gsd_benign_lesion", "gsd_laryng_cancer",
        "gsd_laryngitis", "gsd_laryngeal_dystonia", "gsd_mtd",
        "gsd_precancerous_lesion", "gsd_vocal_fold_paralysis",
        "gsd_parkinsons", "gsd_huntingtons", "gsd_als", "gsd_alz_dementia_mci",
        "gsd_anxiety", "gsd_depression", "gsd_bipolar",
        "gsd_airway_stenosis", "gsd_copd_asthma", "gsd_chronic_cough",
    ]
    available_gsd = [f for f in gsd_flags if f in participants.columns]

    labels = {}
    for pid in participant_ids:
        if pid not in participants.index:
            continue
        row = participants.loc[pid]
        control_col = "gsd_control" if "gsd_control" in participants.columns else "is_control_participant"
        pid_labels = {
            "is_control": int(row[control_col]),
            "cat_voice": int(row["cat_voice"]),
            "cat_neuro": int(row["cat_neuro"]),
            "cat_mood": int(row["cat_mood"]),
            "cat_respiratory": int(row["cat_respiratory"]),
            "age": float(row["age"]) if row["age"] >= 0 else None,
            "phq9_score": float(row["phq9_total"]) if row["phq9_total"] >= 0 else None,
            "gad7_score": float(row["gad7_total"]) if row["gad7_total"] >= 0 else None,
        }
        for flag in available_gsd:
            pid_labels[flag] = int(row[flag])
        labels[pid] = pid_labels
    return labels


def task_stratified_probe_evaluation(
    train_embeddings: dict,
    test_embeddings: dict,
    train_labels: dict,
    test_labels: dict,
    min_participants: int = 10,
) -> dict:
    """Run linear probes separately for each recording task type.

    Instead of averaging all recordings per participant, this groups by
    task type first — e.g., "how well does just the prolonged vowel embedding
    predict neurological conditions?"

    Args:
        train_embeddings, test_embeddings: Output of extract_embeddings.
        train_labels, test_labels: Label dicts from build_label_dicts.
        min_participants: Minimum participants per task type to run probes.

    Returns:
        Dict with keys like "task_stratified/{task_name}/{label}/auroc".
        Also includes a "task_stratified/_summary" key with a nested dict
        {task_name -> {label -> auroc}} for easy heatmap construction.
    """
    metrics = {}
    summary = {}  # task_name -> label -> auroc

    task_names = sorted(set(train_embeddings.get("task_names", [])))
    if not task_names:
        logger.warning("No task_names in embeddings — skipping stratified eval")
        return metrics

    logger.info("Found %d unique task types in train embeddings", len(task_names))
    # Show top task types by recording count
    from collections import Counter
    train_task_counts = Counter(train_embeddings["task_names"])
    for name, count in train_task_counts.most_common(10):
        logger.info("  Task '%s': %d recordings", name, count)

    clf_labels = ["is_control", "cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]

    skipped = 0
    for task_name in task_names:
        # Aggregate per participant using only this task type's recordings
        train_agg = _aggregate_per_participant_by_task(train_embeddings, task_name)
        test_agg = _aggregate_per_participant_by_task(test_embeddings, task_name)

        if len(train_agg) < min_participants or len(test_agg) < min_participants:
            skipped += 1
            continue

        summary[task_name] = {"n_train": len(train_agg), "n_test": len(test_agg)}

        for label_name in clf_labels:
            X_train, y_train = _get_labeled_data(train_agg, train_labels, label_name)
            X_test, y_test = _get_labeled_data(test_agg, test_labels, label_name)

            if len(X_train) < min_participants or len(X_test) < min_participants:
                continue
            if len(set(y_train)) < 2 or len(set(y_test)) < 2:
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X_train_s, y_train)

            y_prob = clf.predict_proba(X_test_s)[:, 1] if len(set(y_test)) == 2 else None
            if y_prob is not None:
                try:
                    auroc = roc_auc_score(y_test, y_prob)
                    metrics[f"task_stratified/{task_name}/{label_name}/auroc"] = auroc
                    summary[task_name][label_name] = auroc
                except ValueError:
                    pass

    metrics["task_stratified/_summary"] = summary
    logger.info("Task-stratified probes: %d task types evaluated, %d skipped (< %d participants)",
                len(summary), skipped, min_participants)
    return metrics


def plot_task_stratified_heatmap(
    stratified_metrics: dict,
    save_path: str | Path,
) -> None:
    """Generate heatmap of AUROC by (recording task type x classification task).

    Args:
        stratified_metrics: Output of task_stratified_probe_evaluation
            (must contain "task_stratified/_summary" key).
        save_path: Path to save figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = stratified_metrics.get("task_stratified/_summary", {})
    if not summary:
        logger.warning("No stratified summary data — skipping heatmap")
        return

    clf_labels = ["is_control", "cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]
    label_display = {
        "is_control": "Control vs Disease",
        "cat_voice": "Voice Disorder",
        "cat_neuro": "Neurological",
        "cat_mood": "Mood Disorder",
        "cat_respiratory": "Respiratory",
    }

    # Filter to task types that have at least one AUROC value
    task_names = [t for t in sorted(summary.keys())
                  if any(l in summary[t] for l in clf_labels)]

    if not task_names:
        logger.warning("No task types with AUROC values — skipping heatmap")
        return

    # Build matrix
    matrix = np.full((len(task_names), len(clf_labels)), np.nan)
    for i, task in enumerate(task_names):
        for j, label in enumerate(clf_labels):
            if label in summary[task]:
                matrix[i, j] = summary[task][label]

    # Truncate long task names for display
    task_display = []
    for t in task_names:
        n = summary[t].get("n_test", "?")
        short = t[:30] + "..." if len(t) > 30 else t
        task_display.append(f"{short} (n={n})")

    # Style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
    })

    fig_height = max(4, 0.4 * len(task_names) + 1.5)
    fig, ax = plt.subplots(figsize=(7, fig_height))

    # Custom colormap: gray for NaN, diverging blue-white-red centered at 0.5
    cmap = plt.cm.RdYlBu_r.copy()
    cmap.set_bad(color="#e0e0e0")

    im = ax.imshow(matrix, cmap=cmap, vmin=0.4, vmax=0.95, aspect="auto")

    # Annotate cells
    for i in range(len(task_names)):
        for j in range(len(clf_labels)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="#999")
            else:
                color = "white" if val > 0.8 or val < 0.45 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7,
                        fontweight="bold" if val > 0.75 else "normal", color=color)

    ax.set_xticks(range(len(clf_labels)))
    ax.set_xticklabels([label_display[l] for l in clf_labels], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels(task_display, fontsize=7)

    fig.colorbar(im, ax=ax, label="AUROC", shrink=0.8, pad=0.02)
    ax.set_title("Linear Probe AUROC by Recording Task Type", fontsize=11, fontweight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Task-stratified heatmap saved to %s", save_path)


def _aggregate_per_participant_by_task(
    embeddings: dict,
    task_name: str,
) -> dict[str, np.ndarray]:
    """Average embeddings per participant, using only recordings of a specific task type."""
    audio = embeddings["audio_embeds"]
    pids = embeddings["participant_ids"]
    tasks = embeddings.get("task_names", [])

    if not tasks:
        return {}

    agg = {}
    for pid in sorted(set(pids)):
        indices = [j for j, (p, t) in enumerate(zip(pids, tasks))
                   if p == pid and t == task_name]
        if indices:
            agg[pid] = audio[indices].mean(axis=0)
    return agg


def plot_umap(
    embeddings: dict,
    participants: pd.DataFrame,
    save_path: str | Path,
) -> None:
    """Generate 2x2 UMAP visualization of audio embeddings.

    Panels: (1) control vs disease, (2) disease category,
    (3) age, (4) site (if available).

    Args:
        embeddings: Output of extract_embeddings.
        participants: DataFrame indexed by record_id.
        save_path: Path to save the PNG figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from umap import UMAP

    # Aggregate per participant
    agg = _aggregate_per_participant(embeddings)
    pids = sorted(agg.keys())
    X = np.array([agg[pid] for pid in pids])

    # Fit UMAP
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(X)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Control vs Disease
    ax = axes[0, 0]
    colors = []
    for pid in pids:
        if pid in participants.index:
            control_col = "gsd_control" if "gsd_control" in participants.columns else "is_control_participant"
            colors.append("tab:blue" if participants.loc[pid, control_col] else "tab:red")
        else:
            colors.append("gray")
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=20, alpha=0.7)
    ax.set_title("Control (blue) vs Disease (red)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Panel 2: Disease categories (multi-label — show primary)
    ax = axes[0, 1]
    cat_names = ["cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]
    cat_colors = ["tab:orange", "tab:purple", "tab:green", "tab:brown"]
    cat_labels = ["Voice", "Neuro", "Mood", "Respiratory"]
    colors = []
    for pid in pids:
        if pid in participants.index:
            row = participants.loc[pid]
            assigned = False
            for cat, col in zip(cat_colors, cat_names):
                if row.get(col, 0) == 1:
                    colors.append(cat)
                    assigned = True
                    break
            if not assigned:
                colors.append("tab:blue")  # control
        else:
            colors.append("gray")
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=20, alpha=0.7)
    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=l)
               for c, l in zip(["tab:blue"] + cat_colors, ["Control"] + cat_labels)]
    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.set_title("Disease Category")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Panel 3: Age (continuous)
    ax = axes[1, 0]
    ages = []
    for pid in pids:
        if pid in participants.index:
            age = participants.loc[pid, "age"]
            ages.append(float(age) if age >= 0 else np.nan)
        else:
            ages.append(np.nan)
    ages = np.array(ages)
    valid = ~np.isnan(ages)
    sc = ax.scatter(coords[valid, 0], coords[valid, 1], c=ages[valid],
                    cmap="viridis", s=20, alpha=0.7)
    if (~valid).any():
        ax.scatter(coords[~valid, 0], coords[~valid, 1], c="gray", s=10, alpha=0.3, label="missing")
    fig.colorbar(sc, ax=ax, label="Age")
    ax.set_title("Age")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Panel 4: Recording task type distribution (pie chart)
    ax = axes[1, 1]
    task_names = embeddings.get("task_names", [])
    if task_names:
        from collections import Counter
        task_counts = Counter(task_names)
        top_tasks = task_counts.most_common(8)
        labels_pie = [t[0][:25] for t in top_tasks]
        sizes = [t[1] for t in top_tasks]
        other = sum(task_counts.values()) - sum(sizes)
        if other > 0:
            labels_pie.append("other")
            sizes.append(other)
        ax.pie(sizes, labels=labels_pie, autopct="%1.0f%%", textprops={"fontsize": 7})
        ax.set_title("Recording Task Distribution")
    else:
        ax.text(0.5, 0.5, "No task data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Recording Tasks")

    fig.suptitle("VoiceFM Embedding Space (UMAP)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("UMAP plot saved to %s", save_path)


def plot_comparison_figures(
    voicefm_metrics: dict,
    hubert_metrics: dict,
    voicefm_curves: dict,
    hubert_curves: dict,
    save_dir: str | Path,
) -> list[Path]:
    """Generate publication-quality comparison figures (VoiceFM vs HuBERT).

    Creates:
        1. AUROC bar chart comparing classification tasks
        2. ROC curves for each binary classification task
        3. Accuracy & F1 grouped bar chart
        4. Regression scatter plots (predicted vs actual)

    Args:
        voicefm_metrics: Metrics dict from VoiceFM linear probes.
        hubert_metrics: Metrics dict from HuBERT baseline probes.
        voicefm_curves: Curves dict from VoiceFM (return_curves=True).
        hubert_curves: Curves dict from HuBERT (return_curves=True).
        save_dir: Directory to save figures.

    Returns:
        List of saved figure paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    save_dir = Path(save_dir)
    saved = []

    # --- Style setup ---
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
    })

    # Okabe-Ito colorblind-safe palette
    C_VOICEFM = "#0072B2"   # blue
    C_HUBERT = "#D55E00"    # vermillion
    C_CHANCE = "#999999"     # gray

    clf_tasks = ["is_control", "cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]
    task_labels = {
        "is_control": "Control\nvs Disease",
        "cat_voice": "Voice\nDisorder",
        "cat_neuro": "Neuro-\nlogical",
        "cat_mood": "Mood\nDisorder",
        "cat_respiratory": "Respiratory\nDisorder",
    }

    # ===== Figure 1: AUROC comparison bar chart =====
    fig, ax = plt.subplots(figsize=(6, 3.5))
    tasks_with_auroc = [t for t in clf_tasks
                        if f"probe/{t}/auroc" in voicefm_metrics
                        and f"probe/{t}/auroc" in hubert_metrics]
    x = np.arange(len(tasks_with_auroc))
    width = 0.35

    vm_aurocs = [voicefm_metrics[f"probe/{t}/auroc"] for t in tasks_with_auroc]
    hb_aurocs = [hubert_metrics[f"probe/{t}/auroc"] for t in tasks_with_auroc]

    bars1 = ax.bar(x - width / 2, vm_aurocs, width, label="VoiceFM",
                   color=C_VOICEFM, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, hb_aurocs, width, label="HuBERT (frozen)",
                   color=C_HUBERT, edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(y=0.5, color=C_CHANCE, linestyle="--", linewidth=0.8, label="Chance")
    ax.set_ylabel("AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels([task_labels[t] for t in tasks_with_auroc])
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Classification Performance: VoiceFM vs Frozen HuBERT")
    fig.tight_layout()
    path = save_dir / "comparison_auroc.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    logger.info("Saved AUROC comparison: %s", path)

    # ===== Figure 2: ROC curves =====
    roc_tasks = [t for t in clf_tasks
                 if t in voicefm_curves and voicefm_curves[t].get("y_prob") is not None
                 and t in hubert_curves and hubert_curves[t].get("y_prob") is not None]

    if roc_tasks:
        ncols = min(len(roc_tasks), 3)
        nrows = (len(roc_tasks) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        if len(roc_tasks) == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        for idx, task in enumerate(roc_tasks):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]

            # VoiceFM ROC
            y_true_v = np.array(voicefm_curves[task]["y_true"])
            y_prob_v = np.array(voicefm_curves[task]["y_prob"])
            fpr_v, tpr_v, _ = roc_curve(y_true_v, y_prob_v)
            auc_v = auc(fpr_v, tpr_v)

            # HuBERT ROC
            y_true_h = np.array(hubert_curves[task]["y_true"])
            y_prob_h = np.array(hubert_curves[task]["y_prob"])
            fpr_h, tpr_h, _ = roc_curve(y_true_h, y_prob_h)
            auc_h = auc(fpr_h, tpr_h)

            ax.plot(fpr_v, tpr_v, color=C_VOICEFM, linewidth=1.5,
                    label=f"VoiceFM (AUC={auc_v:.3f})")
            ax.plot(fpr_h, tpr_h, color=C_HUBERT, linewidth=1.5,
                    label=f"HuBERT (AUC={auc_h:.3f})")
            ax.plot([0, 1], [0, 1], color=C_CHANCE, linestyle="--", linewidth=0.8)

            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            n_test = voicefm_curves[task]["n_test"]
            ax.set_title(f"{task_labels[task].replace(chr(10), ' ')} (n={n_test})")
            ax.legend(loc="lower right", frameon=False, fontsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal")

        # Hide unused axes
        for idx in range(len(roc_tasks), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        fig.suptitle("ROC Curves: VoiceFM vs Frozen HuBERT", fontsize=12, fontweight="bold")
        fig.tight_layout()
        path = save_dir / "comparison_roc_curves.png"
        fig.savefig(str(path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        logger.info("Saved ROC curves: %s", path)

    # ===== Figure 3: Accuracy & F1 grouped bar chart =====
    acc_tasks = [t for t in clf_tasks
                 if f"probe/{t}/accuracy" in voicefm_metrics
                 and f"probe/{t}/accuracy" in hubert_metrics]

    if acc_tasks:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

        # Panel A: Accuracy
        x = np.arange(len(acc_tasks))
        vm_acc = [voicefm_metrics[f"probe/{t}/accuracy"] for t in acc_tasks]
        hb_acc = [hubert_metrics[f"probe/{t}/accuracy"] for t in acc_tasks]

        bars1 = ax1.bar(x - width / 2, vm_acc, width, label="VoiceFM",
                        color=C_VOICEFM, edgecolor="white", linewidth=0.5)
        bars2 = ax1.bar(x + width / 2, hb_acc, width, label="HuBERT (frozen)",
                        color=C_HUBERT, edgecolor="white", linewidth=0.5)
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                         f"{h:.2f}", ha="center", va="bottom", fontsize=6)
        ax1.set_ylabel("Accuracy")
        ax1.set_xticks(x)
        ax1.set_xticklabels([task_labels[t] for t in acc_tasks])
        ax1.set_ylim(0.0, 1.05)
        ax1.legend(loc="upper right", frameon=False, fontsize=7)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.set_title("A  Accuracy", loc="left", fontweight="bold")

        # Panel B: F1 macro
        vm_f1 = [voicefm_metrics.get(f"probe/{t}/f1", 0) for t in acc_tasks]
        hb_f1 = [hubert_metrics.get(f"probe/{t}/f1", 0) for t in acc_tasks]

        bars1 = ax2.bar(x - width / 2, vm_f1, width, label="VoiceFM",
                        color=C_VOICEFM, edgecolor="white", linewidth=0.5)
        bars2 = ax2.bar(x + width / 2, hb_f1, width, label="HuBERT (frozen)",
                        color=C_HUBERT, edgecolor="white", linewidth=0.5)
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                         f"{h:.2f}", ha="center", va="bottom", fontsize=6)
        ax2.set_ylabel("F1 Score (macro)")
        ax2.set_xticks(x)
        ax2.set_xticklabels([task_labels[t] for t in acc_tasks])
        ax2.set_ylim(0.0, 1.05)
        ax2.legend(loc="upper right", frameon=False, fontsize=7)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.set_title("B  F1 Score (macro)", loc="left", fontweight="bold")

        fig.tight_layout()
        path = save_dir / "comparison_acc_f1.png"
        fig.savefig(str(path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        logger.info("Saved accuracy/F1 comparison: %s", path)

    # ===== Figure 4: Regression scatter plots =====
    reg_tasks = [t for t in ["age", "phq9_score", "gad7_score"]
                 if t in voicefm_curves and t in hubert_curves]

    if reg_tasks:
        fig, axes = plt.subplots(1, len(reg_tasks) * 2, figsize=(4 * len(reg_tasks), 3.5))
        if len(reg_tasks) * 2 == 1:
            axes = [axes]
        axes = list(axes)

        reg_labels = {"age": "Age (years)", "phq9_score": "PHQ-9", "gad7_score": "GAD-7"}
        panel_idx = 0

        for task in reg_tasks:
            for model_name, curves, color in [
                ("VoiceFM", voicefm_curves, C_VOICEFM),
                ("HuBERT", hubert_curves, C_HUBERT),
            ]:
                ax = axes[panel_idx]
                y_true = np.array(curves[task]["y_true"])
                y_pred = np.array(curves[task]["y_pred"])

                ax.scatter(y_true, y_pred, alpha=0.5, s=15, color=color, edgecolors="none")

                # Identity line
                lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
                margin = (lims[1] - lims[0]) * 0.05
                lims = [lims[0] - margin, lims[1] + margin]
                ax.plot(lims, lims, color=C_CHANCE, linestyle="--", linewidth=0.8)
                ax.set_xlim(lims)
                ax.set_ylim(lims)

                # Metrics annotation
                rmse_key = f"probe/{task}/rmse"
                r2_key = f"probe/{task}/r2"
                metrics_src = voicefm_metrics if model_name == "VoiceFM" else hubert_metrics
                rmse = metrics_src.get(rmse_key, float("nan"))
                r2 = metrics_src.get(r2_key, float("nan"))
                n = curves[task]["n_test"]
                ax.text(0.05, 0.95,
                        f"RMSE={rmse:.2f}\nR\u00b2={r2:.3f}\nn={n}",
                        transform=ax.transAxes, fontsize=7, va="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

                ax.set_xlabel(f"Actual {reg_labels[task]}")
                ax.set_ylabel(f"Predicted {reg_labels[task]}")
                ax.set_title(f"{model_name}: {reg_labels[task]}")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_aspect("equal")
                panel_idx += 1

        fig.tight_layout()
        path = save_dir / "comparison_regression.png"
        fig.savefig(str(path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        logger.info("Saved regression comparison: %s", path)

    return saved


@torch.no_grad()
def extract_hubert_baseline(
    dataloader: DataLoader,
    device: torch.device,
    backbone: str = "facebook/hubert-base-ls960",
) -> dict:
    """Extract mean-pooled embeddings from a frozen pretrained HuBERT.

    This provides a baseline to compare against VoiceFM fine-tuned embeddings.

    Returns:
        Dict with keys: audio_embeds, participant_ids, recording_ids, task_names.
        audio_embeds are 768-dim mean-pooled HuBERT representations.
    """
    logger.info("Loading pretrained HuBERT for baseline: %s", backbone)
    hubert = HubertModel.from_pretrained(backbone).to(device).eval()

    results = {
        "audio_embeds": [],
        "participant_ids": [],
        "recording_ids": [],
        "task_names": [],
    }

    for batch in dataloader:
        audio = batch["audio_input_values"].to(device)
        mask = batch["attention_mask"].to(device)

        output = hubert(input_values=audio, attention_mask=mask, return_dict=True)
        hidden = output.last_hidden_state  # (B, T, 768)

        # Mean pooling with attention mask
        frame_mask = hubert._get_feature_vector_attention_mask(
            hidden.shape[1], mask,
        )
        frame_mask_f = frame_mask.unsqueeze(-1).float()  # (B, T, 1)
        pooled = (hidden * frame_mask_f).sum(dim=1) / frame_mask_f.sum(dim=1).clamp(min=1)
        pooled = F.normalize(pooled, p=2, dim=-1)

        results["audio_embeds"].append(pooled.cpu())
        results["participant_ids"].extend(batch["participant_id"])
        results["recording_ids"].extend(batch["recording_id"])
        results["task_names"].extend(batch["task_name"])

    results["audio_embeds"] = torch.cat(results["audio_embeds"]).numpy()
    logger.info("Extracted %d baseline HuBERT embeddings (%d-dim)",
                len(results["audio_embeds"]), results["audio_embeds"].shape[1])

    return results


@torch.no_grad()
def extract_hear_baseline(
    dataloader: DataLoader,
    device: torch.device,
    backbone: str = "google/hear-pytorch",
) -> dict:
    """Extract mean-pooled embeddings from a frozen pretrained HeAR.

    This provides a baseline to compare against VoiceFM fine-tuned embeddings
    using HeAR's 512-dim representations without any contrastive training.

    Returns:
        Dict with keys: audio_embeds, participant_ids, recording_ids, task_names.
        audio_embeds are 512-dim mean-pooled HeAR representations.
    """
    from transformers import AutoModel
    from src.models.hear_encoder import (
        MelPCENPreprocessor,
        HEAR_CHUNK_SAMPLES,
        MIN_CHUNK_SAMPLES,
        MAX_CHUNKS_PER_FORWARD,
    )

    logger.info("Loading pretrained HeAR for baseline: %s", backbone)
    hear = AutoModel.from_pretrained(backbone, trust_remote_code=True).to(device).eval()
    preprocessor = MelPCENPreprocessor().to(device)

    results = {
        "audio_embeds": [],
        "participant_ids": [],
        "recording_ids": [],
        "task_names": [],
    }

    for batch in dataloader:
        audio = batch["audio_input_values"].to(device)
        mask = batch["attention_mask"].to(device)
        batch_size = audio.shape[0]

        # Chunk audio into 2-second segments (same logic as HearAudioEncoder)
        all_chunks: list[torch.Tensor] = []
        chunk_counts: list[int] = []
        for i in range(batch_size):
            length = int(mask[i].sum())
            wav = audio[i, :length]
            chunks: list[torch.Tensor] = []
            for start in range(0, max(length, 1), HEAR_CHUNK_SAMPLES):
                remaining = length - start
                if remaining < MIN_CHUNK_SAMPLES and chunks:
                    break
                chunk = torch.zeros(HEAR_CHUNK_SAMPLES, device=device, dtype=audio.dtype)
                copy_len = min(max(remaining, 0), HEAR_CHUNK_SAMPLES)
                if copy_len > 0:
                    chunk[:copy_len] = wav[start : start + copy_len]
                chunks.append(chunk)
            if not chunks:
                chunks.append(torch.zeros(HEAR_CHUNK_SAMPLES, device=device, dtype=audio.dtype))
            all_chunks.extend(chunks)
            chunk_counts.append(len(chunks))

        stacked = torch.stack(all_chunks)

        # Preprocess and run HeAR (disable AMP — mel power spec overflows fp16)
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", enabled=False)
            if stacked.is_cuda
            else nullcontext()
        )
        with amp_ctx:
            spectrograms = preprocessor(stacked.float())
            # Sub-batch for memory safety
            chunk_embeds_list: list[torch.Tensor] = []
            for start_idx in range(0, len(spectrograms), MAX_CHUNKS_PER_FORWARD):
                sub = spectrograms[start_idx : start_idx + MAX_CHUNKS_PER_FORWARD]
                out = hear(pixel_values=sub, return_dict=True)
                chunk_embeds_list.append(out.pooler_output)
            chunk_embeds = torch.cat(chunk_embeds_list, dim=0)

        # Mean-pool chunks per sample
        pooled = torch.zeros(batch_size, chunk_embeds.shape[1], device=device)
        idx = 0
        for i, count in enumerate(chunk_counts):
            pooled[i] = chunk_embeds[idx : idx + count].mean(dim=0)
            idx += count

        # L2-normalize
        pooled = F.normalize(pooled, p=2, dim=-1)

        results["audio_embeds"].append(pooled.cpu())
        results["participant_ids"].extend(batch["participant_id"])
        results["recording_ids"].extend(batch["recording_id"])
        results["task_names"].extend(batch["task_name"])

    results["audio_embeds"] = torch.cat(results["audio_embeds"]).numpy()
    logger.info("Extracted %d baseline HeAR embeddings (%d-dim)",
                len(results["audio_embeds"]), results["audio_embeds"].shape[1])

    return results


def _aggregate_per_participant(embeddings: dict) -> dict[str, np.ndarray]:
    """Average embeddings per participant."""
    audio = embeddings["audio_embeds"]
    pids = embeddings["participant_ids"]

    unique_pids = sorted(set(pids))
    agg = {}
    for pid in unique_pids:
        mask = [j for j, p in enumerate(pids) if p == pid]
        agg[pid] = audio[mask].mean(axis=0)

    return agg


def _get_labeled_data(
    participant_embeds: dict[str, np.ndarray],
    labels: dict,
    label_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Get (X, y) arrays for participants that have the given label."""
    X, y = [], []
    for pid, embed in participant_embeds.items():
        if pid in labels and label_name in labels[pid]:
            val = labels[pid][label_name]
            if val is not None:
                X.append(embed)
                y.append(float(val))

    if not X:
        return np.array([]), np.array([])

    return np.array(X), np.array(y)
