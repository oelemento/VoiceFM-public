#!/usr/bin/env python3
"""Download mPower voice dataset from Synapse.

Downloads:
  1. Demographics table (syn5511429) → demographics.csv
  2. Voice Activity metadata (syn5511444) → voice_metadata.csv
  3. Voice audio files (audio_audio.m4a) → audio_mpower/<healthCode>/<recordId>.m4a

Requires:
  pip install synapseclient
  Synapse auth token in ~/.synapseConfig or SYNAPSE_AUTH_TOKEN env var.

Usage:
    python scripts/download_mpower.py --out-dir data/mpower [--skip-audio] [--max-files N]
"""

import argparse
import csv
import logging
import os
import shutil
from pathlib import Path

import synapseclient
import synapseutils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

VOICE_TABLE = "syn5511444"
DEMOGRAPHICS_TABLE = "syn5511429"


def download_metadata(syn, out_dir: Path):
    """Download demographics and voice metadata tables."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Demographics
    logger.info("Downloading demographics table...")
    results = syn.tableQuery(f"SELECT * FROM {DEMOGRAPHICS_TABLE}")
    df = results.asDataFrame()
    demo_path = out_dir / "demographics.csv"
    df.to_csv(demo_path, index=False)
    logger.info("Saved %d rows to %s", len(df), demo_path)

    # Voice metadata
    logger.info("Downloading voice activity metadata...")
    results = syn.tableQuery(f"SELECT * FROM {VOICE_TABLE}")
    df = results.asDataFrame()
    meta_path = out_dir / "voice_metadata.csv"
    df.to_csv(meta_path, index=False)
    logger.info("Saved %d rows to %s", len(df), meta_path)

    return meta_path


def download_audio(syn, out_dir: Path, max_files: int = 0):
    """Download voice audio files from the Voice Activity table.

    Files are saved as: out_dir/audio/<healthCode>/<recordId>.m4a
    A manifest CSV tracks what's been downloaded.
    """
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "download_manifest.csv"

    # Load existing manifest to resume
    downloaded = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                downloaded.add(row["recordId"])
        logger.info("Resuming: %d files already downloaded", len(downloaded))

    # Query voice table
    logger.info("Querying voice activity table...")
    results = syn.tableQuery(
        f'SELECT recordId, healthCode, "audio_audio.m4a" FROM {VOICE_TABLE}'
    )

    # Download associated files
    logger.info("Downloading audio files...")
    file_map = syn.downloadTableColumns(results, ["audio_audio.m4a"])

    df = results.asDataFrame()
    logger.info("Processing %d recordings...", len(df))

    n_downloaded = 0
    n_skipped = 0

    with open(manifest_path, "a", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=["recordId", "healthCode", "path"])
        if not downloaded:
            writer.writeheader()

        for _, row in df.iterrows():
            record_id = row["recordId"]
            health_code = row["healthCode"]
            file_handle_id = str(int(row["audio_audio.m4a"]))

            if record_id in downloaded:
                n_skipped += 1
                continue

            if file_handle_id not in file_map:
                logger.debug("No file for record %s", record_id)
                continue

            src_path = file_map[file_handle_id]
            dest_dir = audio_dir / health_code
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{record_id}.m4a"

            shutil.copy2(src_path, dest_path)
            writer.writerow({
                "recordId": record_id,
                "healthCode": health_code,
                "path": str(dest_path.relative_to(out_dir)),
            })

            n_downloaded += 1
            if n_downloaded % 1000 == 0:
                logger.info("Downloaded %d files...", n_downloaded)

            if max_files > 0 and n_downloaded >= max_files:
                logger.info("Reached max_files limit (%d)", max_files)
                break

    logger.info(
        "Done: %d downloaded, %d skipped (already existed)", n_downloaded, n_skipped
    )


def main():
    parser = argparse.ArgumentParser(description="Download mPower voice dataset")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/mpower"),
        help="Output directory (default: data/mpower)",
    )
    parser.add_argument(
        "--skip-audio", action="store_true",
        help="Download only metadata, skip audio files",
    )
    parser.add_argument(
        "--max-files", type=int, default=0,
        help="Max audio files to download (0 = all, default: 0)",
    )
    args = parser.parse_args()

    syn = synapseclient.Synapse()
    syn.login(silent=True)

    download_metadata(syn, args.out_dir)

    if not args.skip_audio:
        download_audio(syn, args.out_dir, max_files=args.max_files)
    else:
        logger.info("Skipping audio download (--skip-audio)")


if __name__ == "__main__":
    main()
