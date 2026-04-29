"""
Recording manifest builder for VoiceFM.

Extracts all Recording rows from the Bridge2AI REDCap CSV and produces a
flat manifest (recordings.parquet) mapping each recording to its participant,
task type, duration, and audio filename.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns to extract from Recording rows
_KEEP_COLS = [
    "record_id",
    "recording_id",
    "recording_name",
    "recording_duration",
    "recording_size",
    "recording_microphone",
    "recording_session_id",
]


class RecordingManifest:
    """Builds a per-recording manifest from the REDCap CSV."""

    def process(self, csv_path: str | Path) -> pd.DataFrame:
        """Extract recording rows, clean, and enrich with derived columns.

        Parameters
        ----------
        csv_path : path to the Bridge2AI REDCap CSV export.

        Returns
        -------
        pd.DataFrame with one row per recording.
        """
        csv_path = Path(csv_path)
        logger.info("Loading REDCap CSV: %s", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info("Loaded %d rows x %d columns", *df.shape)

        # ── 1. Filter to Recording rows ────────────────────────────────
        rec = df[df["redcap_repeat_instrument"] == "Recording"].copy()
        logger.info("Recording rows: %d", len(rec))

        # ── 2. Select and rename columns ───────────────────────────────
        available = [c for c in _KEEP_COLS if c in rec.columns]
        manifest = rec[available].copy()

        # ── 3. Derive audio filename ───────────────────────────────────
        manifest["audio_filename"] = manifest["recording_id"] + ".wav"

        # ── 4. Placeholder site column ─────────────────────────────────
        # Will be resolved later by mapping participant_id -> site bucket.
        manifest["site"] = np.nan

        # ── 5. Clean up dtypes ─────────────────────────────────────────
        manifest["recording_duration"] = pd.to_numeric(
            manifest["recording_duration"], errors="coerce"
        ).astype(np.float32)
        manifest["recording_size"] = pd.to_numeric(
            manifest["recording_size"], errors="coerce"
        ).astype(np.float32)

        # Drop rows where recording_id is missing (shouldn't happen but be safe)
        n_before = len(manifest)
        manifest = manifest.dropna(subset=["recording_id"])
        if len(manifest) < n_before:
            logger.warning(
                "Dropped %d rows with missing recording_id", n_before - len(manifest)
            )

        # Reset index for a clean output
        manifest = manifest.reset_index(drop=True)
        logger.info("Final manifest: %d rows x %d columns", *manifest.shape)
        return manifest


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    ROOT = Path(__file__).resolve().parents[2]  # VoiceFM/
    CSV_PATH = ROOT / "data" / "metadata" / (
        "bridge2ai_voice_redcap_data_v2.3.0_2026-02-01T00.00.00.304Z.csv"
    )
    OUT_PATH = ROOT / "data" / "processed" / "recordings.parquet"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    builder = RecordingManifest()
    manifest = builder.process(CSV_PATH)

    # Save
    manifest.to_parquet(OUT_PATH, engine="pyarrow", index=False)
    logger.info("Saved to %s", OUT_PATH)

    # ── Summary stats ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RECORDING MANIFEST SUMMARY")
    print("=" * 60)
    print(f"Shape: {manifest.shape}")
    print(f"Unique participants: {manifest['record_id'].nunique()}")
    print(f"Unique recordings:   {manifest['recording_id'].nunique()}")
    print()

    print("-- Task type distribution (top 15) --")
    task_counts = manifest["recording_name"].value_counts()
    for task, count in task_counts.head(15).items():
        print(f"  {task}: {count}")
    if len(task_counts) > 15:
        print(f"  ... and {len(task_counts) - 15} more task types")
    print()

    print("-- Duration stats (seconds) --")
    dur = manifest["recording_duration"].dropna()
    print(f"  count: {len(dur)}")
    print(f"  mean:  {dur.mean():.1f}")
    print(f"  median:{dur.median():.1f}")
    print(f"  min:   {dur.min():.1f}")
    print(f"  max:   {dur.max():.1f}")
    print(f"  total hours: {dur.sum() / 3600:.1f}")
    print()

    print("-- File size stats (bytes) --")
    sz = manifest["recording_size"].dropna()
    print(f"  count: {len(sz)}")
    print(f"  mean:  {sz.mean():,.0f}")
    print(f"  total GB: {sz.sum() / 1e9:.2f}")
    print()

    print("-- Microphone distribution --")
    mic_counts = manifest["recording_microphone"].value_counts(dropna=False)
    for mic, count in mic_counts.items():
        label = mic if pd.notna(mic) else "<missing>"
        print(f"  {label}: {count}")
    print()

    print("-- Recordings per participant --")
    rpp = manifest.groupby("record_id").size()
    print(f"  mean:   {rpp.mean():.1f}")
    print(f"  median: {rpp.median():.0f}")
    print(f"  min:    {rpp.min()}")
    print(f"  max:    {rpp.max()}")
    print("=" * 60)
