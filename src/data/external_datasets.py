"""PyTorch datasets for external clinical voice data (mPower, SVD, Coswara, VOICED)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.preprocessing import load_and_preprocess, pad_waveform

logger = logging.getLogger(__name__)


class MPowerDataset(Dataset):
    """mPower Parkinson's voice dataset.

    Each item yields audio + labels for multi-task pre-training:
      - is_pd: binary PD vs control
      - age: float (z-scored), NaN if missing
      - sex: int categorical (0=male, 1=female, 2=other)
    """

    DATASET_ID = 0
    DISEASE_COLUMN = "is_pd"

    def __init__(
        self,
        metadata_csv: str | Path,
        audio_dir: str | Path,
        sample_rate: int = 16000,
        max_duration: float = 30,
        task_type_map: dict[str, int] | None = None,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)
        self.task_type_map = task_type_map or {}

        df = pd.read_csv(metadata_csv)
        # Filter to rows with existing audio
        df = df[df["filename"].apply(lambda f: (self.audio_dir / f).exists())]
        df = df.reset_index(drop=True)
        self.metadata = df
        logger.info(f"MPowerDataset: {len(df)} recordings")
        if self.task_type_map:
            logger.info(f"  task_type_map: {self.task_type_map}")

        # Age normalization stats
        ages = df["age"].replace(-1, float("nan")).dropna()
        self.age_mean = float(ages.mean()) if len(ages) > 0 else 0.0
        self.age_std = float(ages.std()) if len(ages) > 0 and ages.std() > 0 else 1.0

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]

        # Load audio
        wav_path = self.audio_dir / row["filename"]
        waveform = load_and_preprocess(wav_path, max_samples=self.max_samples)
        audio_values, attention_mask = pad_waveform(waveform, self.max_samples)

        # Labels
        age_val = float(row["age"])
        if age_val == -1 or pd.isna(age_val):
            age = float("nan")
        else:
            age = (age_val - self.age_mean) / self.age_std

        # Task type ID from recording_type column + task_type_map
        task_type_id = 0
        if self.task_type_map and "recording_type" in self.metadata.columns:
            rec_type = str(row["recording_type"])
            task_type_id = self.task_type_map.get(rec_type, 0)

        result = {
            "audio_values": audio_values,
            "attention_mask": attention_mask,
            "dataset_id": self.DATASET_ID,
            "task_type_id": task_type_id,
            "participant_id": str(row["participant_id"]),
            "labels": {
                "disease": torch.tensor(int(row["is_pd"]), dtype=torch.long),
                "age": torch.tensor(age, dtype=torch.float32),
                "sex": torch.tensor(int(row["sex"]), dtype=torch.long),
            },
        }
        if "recording_id" in self.metadata.columns:
            result["recording_id"] = str(row["recording_id"])
        return result


class SVDDataset(Dataset):
    """Saarbrucken Voice Database for voice pathology detection.

    Each item yields audio + labels:
      - is_pathological: binary pathological vs healthy
      - age: float (z-scored), NaN if missing
      - sex: int categorical (0=male, 1=female, 2=other)
    """

    DATASET_ID = 1
    DISEASE_COLUMN = "is_pathological"

    def __init__(
        self,
        metadata_csv: str | Path,
        audio_dir: str | Path,
        sample_rate: int = 16000,
        max_duration: float = 30,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)

        df = pd.read_csv(metadata_csv)
        df = df[df["filename"].apply(lambda f: (self.audio_dir / f).exists())]
        df = df.reset_index(drop=True)
        self.metadata = df
        logger.info(f"SVDDataset: {len(df)} recordings")

        ages = df["age"].replace(-1, float("nan")).dropna()
        self.age_mean = float(ages.mean()) if len(ages) > 0 else 0.0
        self.age_std = float(ages.std()) if len(ages) > 0 and ages.std() > 0 else 1.0

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]

        wav_path = self.audio_dir / row["filename"]
        waveform = load_and_preprocess(wav_path, max_samples=self.max_samples)
        audio_values, attention_mask = pad_waveform(waveform, self.max_samples)

        age_val = float(row["age"])
        if age_val == -1 or pd.isna(age_val):
            age = float("nan")
        else:
            age = (age_val - self.age_mean) / self.age_std

        return {
            "audio_values": audio_values,
            "attention_mask": attention_mask,
            "dataset_id": self.DATASET_ID,
            "labels": {
                "disease": torch.tensor(int(row["is_pathological"]), dtype=torch.long),
                "age": torch.tensor(age, dtype=torch.float32),
                "sex": torch.tensor(int(row["sex"]), dtype=torch.long),
            },
        }


class CoswaraDataset(Dataset):
    """Coswara COVID-19 voice dataset.

    Each item yields audio + labels:
      - is_covid: binary COVID positive vs healthy
      - age: float (z-scored), NaN if missing
      - sex: int categorical (0=male, 1=female, 2=other)
    """

    DATASET_ID = 2
    DISEASE_COLUMN = "is_covid"

    def __init__(
        self,
        metadata_csv: str | Path,
        audio_dir: str | Path,
        sample_rate: int = 16000,
        max_duration: float = 30,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)

        df = pd.read_csv(metadata_csv)
        df = df[df["filename"].apply(lambda f: (self.audio_dir / f).exists())]
        df = df.reset_index(drop=True)
        self.metadata = df
        logger.info(f"CoswaraDataset: {len(df)} recordings")

        ages = df["age"].replace(-1, float("nan")).dropna()
        self.age_mean = float(ages.mean()) if len(ages) > 0 else 0.0
        self.age_std = float(ages.std()) if len(ages) > 0 and ages.std() > 0 else 1.0

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]

        wav_path = self.audio_dir / row["filename"]
        waveform = load_and_preprocess(wav_path, max_samples=self.max_samples)
        audio_values, attention_mask = pad_waveform(waveform, self.max_samples)

        age_val = float(row["age"])
        if age_val == -1 or pd.isna(age_val):
            age = float("nan")
        else:
            age = (age_val - self.age_mean) / self.age_std

        return {
            "audio_values": audio_values,
            "attention_mask": attention_mask,
            "dataset_id": self.DATASET_ID,
            "labels": {
                "disease": torch.tensor(int(row["is_covid"]), dtype=torch.long),
                "age": torch.tensor(age, dtype=torch.float32),
                "sex": torch.tensor(int(row["sex"]), dtype=torch.long),
            },
        }


class VOICEDDataset(Dataset):
    """VOICED (PhysioNet) voice pathology dataset.

    Each item yields audio + labels:
      - is_pathological: binary pathological vs healthy
      - age: float (z-scored), NaN if missing
      - sex: int categorical (0=male, 1=female, 2=other)
    """

    DATASET_ID = 3
    DISEASE_COLUMN = "is_pathological"

    def __init__(
        self,
        metadata_csv: str | Path,
        audio_dir: str | Path,
        sample_rate: int = 16000,
        max_duration: float = 30,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)

        df = pd.read_csv(metadata_csv)
        df = df[df["filename"].apply(lambda f: (self.audio_dir / f).exists())]
        df = df.reset_index(drop=True)
        self.metadata = df
        logger.info(f"VOICEDDataset: {len(df)} recordings")

        ages = df["age"].replace(-1, float("nan")).dropna()
        self.age_mean = float(ages.mean()) if len(ages) > 0 else 0.0
        self.age_std = float(ages.std()) if len(ages) > 0 and ages.std() > 0 else 1.0

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]

        wav_path = self.audio_dir / row["filename"]
        waveform = load_and_preprocess(wav_path, max_samples=self.max_samples)
        audio_values, attention_mask = pad_waveform(waveform, self.max_samples)

        age_val = float(row["age"])
        if age_val == -1 or pd.isna(age_val):
            age = float("nan")
        else:
            age = (age_val - self.age_mean) / self.age_std

        return {
            "audio_values": audio_values,
            "attention_mask": attention_mask,
            "dataset_id": self.DATASET_ID,
            "labels": {
                "disease": torch.tensor(int(row["is_pathological"]), dtype=torch.long),
                "age": torch.tensor(age, dtype=torch.float32),
                "sex": torch.tensor(int(row["sex"]), dtype=torch.long),
            },
        }


class FigsharePDDataset(Dataset):
    """Figshare Parkinson's Disease voice dataset.

    Each item yields audio + labels:
      - label: binary PD (1) vs healthy control (0)
      - age: float (z-scored)
      - sex: int categorical (0=male, 1=female)
    """

    DATASET_ID = 4
    DISEASE_COLUMN = "label"

    def __init__(
        self,
        metadata_csv: str | Path,
        audio_dir: str | Path,
        sample_rate: int = 16000,
        max_duration: float = 30,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)

        df = pd.read_csv(metadata_csv)
        # filepath column is relative to dataset root (e.g., audio/HC_AH/file.wav)
        df = df[df["filepath"].apply(lambda f: (self.audio_dir / f).exists())]
        df = df.reset_index(drop=True)
        self.metadata = df
        logger.info(f"FigsharePDDataset: {len(df)} recordings")

        ages = df["age"].replace(-1, float("nan")).dropna()
        self.age_mean = float(ages.mean()) if len(ages) > 0 else 0.0
        self.age_std = float(ages.std()) if len(ages) > 0 and ages.std() > 0 else 1.0

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]

        wav_path = self.audio_dir / row["filepath"]
        waveform = load_and_preprocess(wav_path, max_samples=self.max_samples)
        audio_values, attention_mask = pad_waveform(waveform, self.max_samples)

        age_val = float(row["age"])
        if age_val == -1 or pd.isna(age_val):
            age = float("nan")
        else:
            age = (age_val - self.age_mean) / self.age_std

        sex_map = {"M": 0, "F": 1}
        sex = sex_map.get(str(row["sex"]).strip(), 2)

        return {
            "audio_values": audio_values,
            "attention_mask": attention_mask,
            "dataset_id": self.DATASET_ID,
            "labels": {
                "disease": torch.tensor(int(row["label"]), dtype=torch.long),
                "age": torch.tensor(age, dtype=torch.float32),
                "sex": torch.tensor(sex, dtype=torch.long),
            },
        }


class NeuroVozDataset(Dataset):
    """NeuroVoz Parkinson's Disease voice dataset (Spanish).

    108 participants (55 HC, 53 PD), ~2900 recordings across vowels,
    sentence reading, DDK, and spontaneous speech tasks.

    Each item yields audio + labels:
      - label: binary PD (1) vs healthy control (0)
      - age: from metadata
      - sex: 1=male, 0=female
    """

    DATASET_ID = 7
    DISEASE_COLUMN = "label"

    def __init__(
        self,
        metadata_csv: str | Path,
        audio_dir: str | Path,
        sample_rate: int = 16000,
        max_duration: float = 30,
        task_category: str | None = None,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)

        df = pd.read_csv(metadata_csv)
        if task_category:
            df = df[df["task_category"] == task_category]
        df = df[df["filename"].apply(lambda f: (self.audio_dir / f).exists())]
        df = df.reset_index(drop=True)
        self.metadata = df
        cat_str = f" ({task_category})" if task_category else ""
        logger.info(f"NeuroVozDataset{cat_str}: {len(df)} recordings, {df['participant_id'].nunique()} participants")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]

        wav_path = self.audio_dir / row["filename"]
        waveform = load_and_preprocess(wav_path, max_samples=self.max_samples)
        audio_values, attention_mask = pad_waveform(waveform, self.max_samples)

        age = float(row["age"]) if pd.notna(row["age"]) else float("nan")
        sex = int(row["sex"]) if pd.notna(row["sex"]) else 2

        return {
            "audio_values": audio_values,
            "attention_mask": attention_mask,
            "dataset_id": self.DATASET_ID,
            "participant_id": str(row["participant_id"]),
            "labels": {
                "disease": torch.tensor(int(row["label"]), dtype=torch.long),
                "age": torch.tensor(age, dtype=torch.float32),
                "sex": torch.tensor(sex, dtype=torch.long),
            },
        }


class MDVRKCLDataset(Dataset):
    """MDVR-KCL Parkinson's Disease voice dataset.

    Each item yields audio + labels:
      - label: binary PD (1) vs healthy control (0)
      - age: NaN (not available in metadata)
      - sex: 2 (unknown, not available in metadata)
    """

    DATASET_ID = 5
    DISEASE_COLUMN = "label"

    def __init__(
        self,
        metadata_csv: str | Path,
        audio_dir: str | Path,
        sample_rate: int = 16000,
        max_duration: float = 30,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)

        df = pd.read_csv(metadata_csv)
        df = df[df["filename"].apply(lambda f: (self.audio_dir / f).exists())]
        df = df.reset_index(drop=True)
        self.metadata = df
        logger.info(f"MDVRKCLDataset: {len(df)} recordings")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]

        wav_path = self.audio_dir / row["filename"]
        waveform = load_and_preprocess(wav_path, max_samples=self.max_samples)
        audio_values, attention_mask = pad_waveform(waveform, self.max_samples)

        return {
            "audio_values": audio_values,
            "attention_mask": attention_mask,
            "dataset_id": self.DATASET_ID,
            "labels": {
                "disease": torch.tensor(int(row["label"]), dtype=torch.long),
                "age": torch.tensor(float("nan"), dtype=torch.float32),
                "sex": torch.tensor(2, dtype=torch.long),
            },
        }


class PVQDDataset(Dataset):
    """PVQD (Perceptual Voice Qualities Database) for CAPE-V evaluation.

    Each item yields audio + CAPE-V / GRBAS ratings:
      - is_pathological: binary pathological vs healthy
      - age: float (z-scored), -1 if missing
      - sex: int categorical (0=male, 1=female, 2=unknown)
      - capev_*: CAPE-V ratings (0-100 VAS)
      - grbas_*: GRBAS ratings (0-3 ordinal, averaged)
    """

    DATASET_ID = 6
    DISEASE_COLUMN = "is_pathological"

    CAPEV_COLS = [
        "capev_severity", "capev_roughness", "capev_breathiness",
        "capev_strain", "capev_pitch", "capev_loudness",
    ]
    GRBAS_COLS = [
        "grbas_grade", "grbas_roughness", "grbas_breathiness",
        "grbas_asthenia", "grbas_strain",
    ]

    def __init__(
        self,
        metadata_csv: str | Path,
        audio_dir: str | Path,
        sample_rate: int = 16000,
        max_duration: float = 30,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)

        df = pd.read_csv(metadata_csv)
        df = df[df["filename"].apply(lambda f: (self.audio_dir / f).exists())]
        # Filter to rows with known pathological status for disease classification
        df = df[df["is_pathological"].isin([0, 1])]
        df = df.reset_index(drop=True)
        self.metadata = df
        logger.info(f"PVQDDataset: {len(df)} recordings")

        ages = df["age"].replace(-1, float("nan")).dropna()
        self.age_mean = float(ages.mean()) if len(ages) > 0 else 0.0
        self.age_std = float(ages.std()) if len(ages) > 0 and ages.std() > 0 else 1.0

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]

        wav_path = self.audio_dir / row["filename"]
        waveform = load_and_preprocess(wav_path, max_samples=self.max_samples)
        audio_values, attention_mask = pad_waveform(waveform, self.max_samples)

        age_val = float(row["age"])
        if age_val == -1 or pd.isna(age_val):
            age = float("nan")
        else:
            age = (age_val - self.age_mean) / self.age_std

        # CAPE-V and GRBAS ratings
        capev = {}
        for col in self.CAPEV_COLS:
            val = row.get(col)
            capev[col] = float(val) if pd.notna(val) else float("nan")

        grbas = {}
        for col in self.GRBAS_COLS:
            val = row.get(col)
            grbas[col] = float(val) if pd.notna(val) else float("nan")

        return {
            "audio_values": audio_values,
            "attention_mask": attention_mask,
            "dataset_id": self.DATASET_ID,
            "participant_id": str(row["participant_id"]),
            "labels": {
                "disease": torch.tensor(int(row["is_pathological"]), dtype=torch.long),
                "age": torch.tensor(age, dtype=torch.float32),
                "sex": torch.tensor(int(row["sex"]), dtype=torch.long),
            },
            "capev": {k: torch.tensor(v, dtype=torch.float32) for k, v in capev.items()},
            "grbas": {k: torch.tensor(v, dtype=torch.float32) for k, v in grbas.items()},
        }


class CombinedExternalDataset(Dataset):
    """Wraps multiple external datasets, preserving dataset_id for per-dataset heads."""

    def __init__(self, datasets: list[Dataset]) -> None:
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def _locate(self, idx: int) -> tuple[int, int]:
        """Return (dataset_index, local_index) for a global index."""
        offset = 0
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                return i, idx - offset
            offset = size
        raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

    def __getitem__(self, idx: int) -> dict:
        ds_idx, local_idx = self._locate(idx)
        return self.datasets[ds_idx][local_idx]

    def get_stratification_labels(self) -> list[str]:
        """Return stratification labels from metadata without loading audio."""
        labels = []
        for ds in self.datasets:
            ds_id = ds.DATASET_ID
            for disease_val in ds.metadata[ds.DISEASE_COLUMN].astype(int):
                labels.append(f"{ds_id}_{disease_val}")
        return labels


class ExternalMultitaskDataset(Dataset):
    """Wraps an external dataset to output VoiceFMDataset-compatible batches.

    Maps external disease labels to multitask clinical_features dict.
    All unmapped features are set to NaN (masked by MultiTaskLoss).

    Args:
        external_dataset: An SVDDataset, CoswaraDataset, etc.
        target_key: Which clinical feature this dataset's disease maps to
            (e.g., "cat_voice" for SVD, "cat_respiratory" for Coswara).
        feature_config: Feature names dict from ClinicalFeatureProcessor.get_feature_names().
        task_type_id: Integer task type ID for these recordings (default 0 = unknown).
    """

    def __init__(
        self,
        external_dataset: Dataset,
        target_key: str,
        feature_config: dict,
        task_type_id: int = 0,
    ) -> None:
        self.dataset = external_dataset
        self.target_key = target_key
        self.feature_config = feature_config
        self.task_type_id = task_type_id

        # Build template of NaN/zero tensors for all clinical features
        self._nan_clinical = {}
        for col in feature_config.get("binary", []):
            self._nan_clinical[col] = torch.tensor(float("nan"), dtype=torch.float32)
        for col in feature_config.get("continuous", []):
            self._nan_clinical[col] = torch.tensor(float("nan"), dtype=torch.float32)
        for col in feature_config.get("categorical", {}).keys():
            self._nan_clinical[col] = torch.tensor(float("nan"), dtype=torch.float32)

        # Multi-task model task keys (from multitask_model.yaml input_keys)
        for key in ["is_control_participant", "cat_neuro", "cat_voice", "cat_mood",
                     "cat_respiratory", "age", "phq9_total", "gad7_total", "vhi10_total"]:
            if key not in self._nan_clinical:
                self._nan_clinical[key] = torch.tensor(float("nan"), dtype=torch.float32)

        # disease_categories for contrastive aux loss (4-dim multi-label)
        self._nan_clinical["disease_categories"] = torch.tensor(
            [float("nan")] * 4, dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        # Build clinical features dict: all NaN except the mapped label
        clinical = {k: v.clone() for k, v in self._nan_clinical.items()}
        disease_label = item["labels"]["disease"].float()
        clinical[self.target_key] = disease_label

        # Map age if available
        age_val = item["labels"]["age"]
        if not torch.isnan(age_val):
            clinical["age"] = age_val

        return {
            "audio_input_values": item["audio_values"],
            "attention_mask": item["attention_mask"],
            "task_type_id": torch.tensor(self.task_type_id, dtype=torch.long),
            "site_id": torch.tensor(-1, dtype=torch.long),
            "clinical_features": clinical,
            "recording_id": f"ext_{item['dataset_id']}_{idx}",
            "participant_id": f"ext_{item['dataset_id']}_{idx}",
            "task_name": "external",
        }


def external_collate_fn(batch: list[dict]) -> dict:
    """Collate function for external dataset batches."""
    result = {
        "audio_values": torch.stack([b["audio_values"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "dataset_ids": torch.tensor([b["dataset_id"] for b in batch], dtype=torch.long),
        "labels": {
            "disease": torch.stack([b["labels"]["disease"] for b in batch]),
            "age": torch.stack([b["labels"]["age"] for b in batch]),
            "sex": torch.stack([b["labels"]["sex"] for b in batch]),
        },
    }
    if "participant_id" in batch[0]:
        result["participant_ids"] = [b["participant_id"] for b in batch]
    if "task_type_id" in batch[0]:
        result["task_type_ids"] = torch.tensor(
            [b["task_type_id"] for b in batch], dtype=torch.long
        )
    if "recording_id" in batch[0]:
        result["recording_ids"] = [b["recording_id"] for b in batch]
    return result
