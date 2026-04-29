"""PyTorch dataset for VoiceFM audio + clinical data."""

import torch
from torch.utils.data import Dataset
import pandas as pd
import logging
from pathlib import Path

from src.utils.preprocessing import load_and_preprocess, pad_waveform, MAX_SAMPLES

logger = logging.getLogger(__name__)


class VoiceFMDataset(Dataset):
    """Dataset that yields (audio, clinical_features, metadata) tuples.

    Each item corresponds to a single recording. Clinical features are
    looked up by participant ID from the participant table.

    Args:
        recording_manifest: DataFrame with columns: recording_id, record_id,
            recording_name, recording_duration, wav_filename
        participant_table: DataFrame indexed by record_id with clinical features
        audio_dir: Root directory containing WAV files (flat or site-organized)
        task_type_map: Dict mapping recording_name -> integer task type ID
        feature_config: Dict with keys 'binary', 'continuous', 'categorical'
            listing the column names in participant_table
        max_samples: Maximum audio length in samples
        cache_dir: Optional directory for caching preprocessed tensors
    """

    def __init__(
        self,
        recording_manifest: pd.DataFrame,
        participant_table: pd.DataFrame,
        audio_dir: str | Path,
        task_type_map: dict[str, int],
        feature_config: dict[str, list[str]],
        max_samples: int = MAX_SAMPLES,
        cache_dir: str | Path | None = None,
        age_mean: float | None = None,
        age_std: float | None = None,
        site_mapping: dict[str, int] | None = None,
    ):
        self.recordings = recording_manifest.reset_index(drop=True)
        self.participants = participant_table
        self.audio_dir = Path(audio_dir)
        self.task_type_map = task_type_map
        self.feature_config = feature_config
        self.max_samples = max_samples
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.site_mapping = site_mapping or {}

        # Age normalization stats (z-score)
        if age_mean is not None and age_std is not None:
            self.age_mean = age_mean
            self.age_std = age_std
        elif "age" in participant_table.columns:
            ages = participant_table["age"].replace(-1, float("nan")).dropna()
            self.age_mean = ages.mean()
            self.age_std = ages.std() if ages.std() > 0 else 1.0
        else:
            self.age_mean = 0.0
            self.age_std = 1.0

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Build participant ID to row index mapping
        self._pid_index = {
            pid: idx for idx, pid in enumerate(self.participants.index)
        }

        # Precompute participant tensors
        self._participant_tensors = self._build_participant_tensors()

    def _build_participant_tensors(self) -> dict[str, dict[str, torch.Tensor]]:
        """Convert participant table to per-feature scalar tensors.

        The model's FeatureTokenizer expects individual named features,
        so we store each feature as a scalar tensor keyed by column name.
        """
        tensors = {}
        binary_cols = self.feature_config.get("binary", [])
        continuous_cols = self.feature_config.get("continuous", [])
        categorical_cols = list(self.feature_config.get("categorical", {}).keys())

        for pid in self.participants.index:
            row = self.participants.loc[pid]
            t = {}

            for col in binary_cols:
                t[col] = torch.tensor(int(row[col]), dtype=torch.long)

            for col in continuous_cols:
                val = float(row[col])
                # -1 sentinel → NaN for the model's masking logic
                t[col] = torch.tensor(float("nan") if val == -1 else val, dtype=torch.float32)

            for col in categorical_cols:
                t[col] = torch.tensor(int(row[col]), dtype=torch.long)

            # Disease category labels for auxiliary loss (multi-label)
            disease_cols = ["cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]
            if all(c in self.participants.columns for c in disease_cols):
                t["disease_categories"] = torch.tensor(
                    row[disease_cols].values.astype(float), dtype=torch.float32
                )

            # Age for auxiliary regression loss (z-score normalized)
            if "age" in self.participants.columns:
                age_val = float(row["age"])
                if age_val == -1 or pd.isna(age_val):
                    t["age"] = torch.tensor(float("nan"), dtype=torch.float32)
                else:
                    t["age"] = torch.tensor(
                        (age_val - self.age_mean) / self.age_std, dtype=torch.float32
                    )

            tensors[pid] = t

        return tensors

    def __len__(self) -> int:
        return len(self.recordings)

    def __getitem__(self, idx: int) -> dict:
        row = self.recordings.iloc[idx]
        recording_id = row["recording_id"]
        participant_id = row["record_id"]
        task_name = row["recording_name"]

        # Load audio
        wav_path = self.audio_dir / f"{recording_id}.wav"
        cache_path = self.cache_dir / f"{recording_id}.pt" if self.cache_dir else None

        if cache_path and cache_path.exists():
            waveform = torch.load(cache_path, weights_only=True)
        else:
            if not wav_path.exists():
                # Return zeros if file missing (will be masked)
                waveform = torch.zeros(self.max_samples)
                logger.warning(f"Missing audio: {wav_path}")
            else:
                waveform = load_and_preprocess(wav_path, max_samples=self.max_samples)

            if cache_path:
                torch.save(waveform, cache_path)

        # Pad to max_samples
        audio_values, attention_mask = pad_waveform(waveform, self.max_samples)

        # Task type
        task_type_id = self.task_type_map.get(task_name, 0)

        # Clinical features
        clinical = self._participant_tensors.get(participant_id, {})

        # Site ID (for DANN)
        site_id = self.site_mapping.get(recording_id, -1)

        return {
            "audio_input_values": audio_values,
            "attention_mask": attention_mask,
            "task_type_id": torch.tensor(task_type_id, dtype=torch.long),
            "site_id": torch.tensor(site_id, dtype=torch.long),
            "clinical_features": clinical,
            "recording_id": recording_id,
            "participant_id": participant_id,
            "task_name": task_name,
        }


def voicefm_collate_fn(batch: list[dict]) -> dict:
    """Custom collate function that handles the nested clinical features dict."""
    result = {
        "audio_input_values": torch.stack([b["audio_input_values"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "task_type_id": torch.stack([b["task_type_id"] for b in batch]),
        "site_id": torch.stack([b["site_id"] for b in batch]),
        "recording_id": [b["recording_id"] for b in batch],
        "participant_id": [b["participant_id"] for b in batch],
        "task_name": [b["task_name"] for b in batch],
    }

    # Stack clinical features: use union of keys across batch items,
    # filling missing keys with NaN tensors (masked by MultiTaskLoss).
    # This handles mixed B2AI / external dataset batches.
    all_keys: set[str] = set()
    for b in batch:
        all_keys.update(b["clinical_features"].keys())

    result["clinical_features"] = {}
    for key in sorted(all_keys):
        # Find a reference tensor for shape/dtype
        ref = next(b["clinical_features"][key] for b in batch if key in b["clinical_features"])
        tensors = []
        for b in batch:
            if key in b["clinical_features"]:
                tensors.append(b["clinical_features"][key])
            else:
                tensors.append(torch.full_like(ref, float("nan")))
        result["clinical_features"][key] = torch.stack(tensors)

    return result


def build_task_type_map(recording_manifest: pd.DataFrame) -> dict[str, int]:
    """Build mapping from recording task names to integer IDs.

    ID 0 is reserved for unknown task types.
    """
    task_names = sorted(recording_manifest["recording_name"].dropna().unique())
    return {name: i + 1 for i, name in enumerate(task_names)}
