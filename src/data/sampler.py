"""Participant-aware batch sampler for contrastive training."""

import logging
import math
import random
import torch
from torch.utils.data import Sampler
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


def _derive_disease_category_labels(participant_table: pd.DataFrame) -> pd.Series:
    """Build compact disease-category labels for stratification.

    Produces one label per participant:
    - control
    - voice / neuro / mood / respiratory (single-category cases)
    - multi (multiple categories positive)
    - unknown (no disease flags and not control)
    """
    required = ["cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]
    if not all(col in participant_table.columns for col in required):
        logger.warning(
            "Missing disease category columns for stratification (%s). Falling back to 'unknown'.",
            ", ".join(required),
        )
        return pd.Series("unknown", index=participant_table.index)

    label_map = {
        "cat_voice": "voice",
        "cat_neuro": "neuro",
        "cat_mood": "mood",
        "cat_respiratory": "respiratory",
    }
    labels = []
    control_col = "gsd_control" if "gsd_control" in participant_table.columns else "is_control_participant"
    is_control = participant_table.get(
        control_col,
        pd.Series(0, index=participant_table.index),
    ).fillna(0).astype(int)
    cats = participant_table[required].fillna(0).astype(int)

    for pid in participant_table.index:
        row = cats.loc[pid]
        positives = [label_map[col] for col in required if int(row[col]) == 1]
        if len(positives) == 0 and int(is_control.loc[pid]) == 1:
            labels.append("control")
        elif len(positives) == 1:
            labels.append(positives[0])
        elif len(positives) > 1:
            labels.append("multi")
        else:
            labels.append("unknown")

    return pd.Series(labels, index=participant_table.index)


def build_participant_strata(
    participant_table: pd.DataFrame,
    stratify_col: str | None = None,
) -> pd.Series | None:
    """Return per-participant labels used for split/sampler stratification."""
    if stratify_col is None:
        return None

    if stratify_col in participant_table.columns:
        strata = participant_table[stratify_col]
    elif stratify_col == "disease_category":
        strata = _derive_disease_category_labels(participant_table)
    else:
        logger.warning(
            "Unknown stratify_col='%s'; available columns include: %s. Using random split.",
            stratify_col,
            ", ".join(participant_table.columns[:12]),
        )
        return None

    return strata.fillna("unknown").astype(str)


def _collapse_rare_strata(
    labels: np.ndarray,
    min_count: int = 2,
    rare_label: str = "other",
) -> np.ndarray:
    """Collapse labels with very small counts into a shared bucket.

    This avoids train_test_split(stratify=...) failures when one class has
    <2 members.
    """
    ser = pd.Series(labels, dtype="object")
    counts = ser.value_counts(dropna=False)
    rare = set(counts[counts < min_count].index.tolist())
    if not rare:
        return labels

    # If rare labels would collapse to a singleton "other", merge into the
    # most common existing label instead.
    non_rare = counts[counts >= min_count]
    if non_rare.empty:
        target_label = rare_label
    else:
        target_label = non_rare.index[0]

    collapsed = np.array([target_label if x in rare else x for x in labels], dtype=object)
    logger.warning(
        "Collapsed rare strata %s into '%s' for robust stratified splitting.",
        sorted(str(v) for v in rare),
        target_label,
    )
    return collapsed


class ParticipantBatchSampler(Sampler[list[int]]):
    """Samples batches where each participant appears at most once.

    For each batch:
    1. Sample N participants (stratified by disease category)
    2. For each participant, sample 1 random recording
    3. Return indices into the recording manifest

    Args:
        recording_manifest: DataFrame with record_id and recording_name columns
        participant_categories: Series mapping record_id -> primary disease category
            (for stratification)
        batch_size: Number of participants per batch
        recordings_per_participant: How many recordings to sample per participant
        task_stratify: If True, try to include diverse task types in each batch
        seed: Random seed
        drop_last: Drop the last incomplete batch
    """

    def __init__(
        self,
        recording_manifest: pd.DataFrame,
        participant_categories: pd.Series | None = None,
        batch_size: int = 64,
        recordings_per_participant: int = 1,
        task_stratify: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.batch_size = batch_size
        self.recordings_per_participant = recordings_per_participant
        self.task_stratify = task_stratify
        self.drop_last = drop_last
        self.rng = random.Random(seed)

        # Build participant -> recording indices mapping
        # Use positional indices (0-based) so they work with dataset.__getitem__(iloc)
        self.participant_recordings: dict[str, list[int]] = defaultdict(list)
        for pos_idx, (_, row) in enumerate(recording_manifest.iterrows()):
            self.participant_recordings[row["record_id"]].append(pos_idx)

        self.participant_ids = list(self.participant_recordings.keys())

        # Task type info for stratification
        if task_stratify:
            self.recording_tasks: dict[int, str] = {}
            for pos_idx, (_, row) in enumerate(recording_manifest.iterrows()):
                self.recording_tasks[pos_idx] = row.get("recording_name", "unknown")

        # Category stratification
        if participant_categories is not None and not isinstance(participant_categories, pd.Series):
            participant_categories = pd.Series(participant_categories)
        self.categories = participant_categories.fillna("unknown").astype(str) if participant_categories is not None else None
        if self.categories is not None:
            self.category_participants: dict[str, list[str]] = defaultdict(list)
            for pid in self.participant_ids:
                cat = self.categories.get(pid, "unknown")
                self.category_participants[cat].append(pid)

    def _build_participant_order(self) -> list[str]:
        """Create per-epoch participant order with optional category interleaving."""
        if self.categories is None:
            participants = self.participant_ids.copy()
            self.rng.shuffle(participants)
            return participants

        # Shuffle within category, then interleave categories round-robin.
        buckets: dict[str, list[str]] = {}
        for cat, pids in self.category_participants.items():
            shuffled = pids.copy()
            self.rng.shuffle(shuffled)
            buckets[cat] = shuffled

        category_order = list(buckets.keys())
        self.rng.shuffle(category_order)

        participants = []
        while True:
            added = False
            for cat in category_order:
                bucket = buckets[cat]
                if bucket:
                    participants.append(bucket.pop())
                    added = True
            if not added:
                break
            self.rng.shuffle(category_order)

        return participants

    def _sample_participant_recordings(self, pid: str) -> list[int]:
        """Sample recordings for one participant, optionally diversifying tasks."""
        recordings = self.participant_recordings[pid]
        n = min(self.recordings_per_participant, len(recordings))
        if n <= 1:
            return self.rng.sample(recordings, n)

        if not self.task_stratify:
            return self.rng.sample(recordings, n)

        # Prefer one recording per distinct task before repeating task types.
        by_task: dict[str, list[int]] = defaultdict(list)
        for rec_idx in recordings:
            task = self.recording_tasks.get(rec_idx, "unknown")
            by_task[task].append(rec_idx)

        task_names = list(by_task.keys())
        self.rng.shuffle(task_names)

        sampled = []
        for task in task_names:
            if len(sampled) >= n:
                break
            sampled.append(self.rng.choice(by_task[task]))

        if len(sampled) < n:
            remaining = [idx for idx in recordings if idx not in sampled]
            self.rng.shuffle(remaining)
            sampled.extend(remaining[: n - len(sampled)])

        return sampled

    def __iter__(self):
        participants = self._build_participant_order()

        # Create batches
        batches = []
        for i in range(0, len(participants), self.batch_size):
            batch_pids = participants[i : i + self.batch_size]
            if self.drop_last and len(batch_pids) < self.batch_size:
                break

            # For each participant, sample recordings
            batch_indices = []
            for pid in batch_pids:
                batch_indices.extend(self._sample_participant_recordings(pid))

            batches.append(batch_indices)

        self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.participant_ids) // self.batch_size
        return math.ceil(len(self.participant_ids) / self.batch_size)


def create_participant_splits(
    participant_table: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_col: str | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Split participants into train/val/test sets.

    Args:
        participant_table: DataFrame indexed by record_id
        train_ratio, val_ratio, test_ratio: Split ratios
        seed: Random seed
        stratify_col: Column to stratify by (optional)

    Returns:
        (train_ids, val_ids, test_ids) lists of participant IDs
    """
    rng = np.random.RandomState(seed)
    pids = participant_table.index.tolist()

    strata = build_participant_strata(participant_table, stratify_col)

    if strata is not None:
        from sklearn.model_selection import train_test_split

        # Convert to plain NumPy arrays to avoid pandas ArrowExtension indexing
        # issues inside sklearn's _safe_indexing on some cluster environments.
        pid_arr = np.asarray(pids, dtype=object)
        labels = strata.reindex(pids).fillna("unknown").astype(str).to_numpy(dtype=object)
        labels = _collapse_rare_strata(labels, min_count=2, rare_label="other")

        try:
            # First split: train vs (val + test)
            train_ids, temp_ids, _, temp_labels = train_test_split(
                pid_arr, labels, test_size=(val_ratio + test_ratio),
                random_state=seed, stratify=labels,
            )

            # Second split: val vs test
            relative_test = test_ratio / (val_ratio + test_ratio)
            temp_ids = np.asarray(temp_ids, dtype=object)
            temp_labels = np.asarray(temp_labels, dtype=object)
            temp_labels = _collapse_rare_strata(temp_labels, min_count=2, rare_label="other")
            val_ids, test_ids = train_test_split(
                temp_ids, test_size=relative_test,
                random_state=seed, stratify=temp_labels,
            )
            train_ids = train_ids.tolist()
            val_ids = val_ids.tolist()
            test_ids = test_ids.tolist()
        except ValueError as exc:
            logger.warning(
                "Stratified split failed for stratify_col='%s' (%s). Falling back to random split.",
                stratify_col, exc,
            )
            rng.shuffle(pids)
            n = len(pids)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_ids = pids[:n_train]
            val_ids = pids[n_train : n_train + n_val]
            test_ids = pids[n_train + n_val :]
        except Exception as exc:
            logger.warning(
                "Unexpected split failure for stratify_col='%s' (%s). Falling back to random split.",
                stratify_col, exc,
            )
            rng.shuffle(pids)
            n = len(pids)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_ids = pids[:n_train]
            val_ids = pids[n_train : n_train + n_val]
            test_ids = pids[n_train + n_val :]
    else:
        rng.shuffle(pids)
        n = len(pids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_ids = pids[:n_train]
        val_ids = pids[n_train : n_train + n_val]
        test_ids = pids[n_train + n_val :]

    return train_ids, val_ids, test_ids
