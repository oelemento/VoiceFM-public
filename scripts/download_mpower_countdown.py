#!/usr/bin/env python3
"""Download mPower countdown audio files from Synapse.

Downloads audio_countdown.m4a files that were not included in the original
download (which only grabbed audio_audio.m4a sustained phonation).

Uses individual file handle downloads instead of bulk downloadTableColumns,
which gets stuck in a retry loop for the countdown column.

Files are saved as: data/mpower/audio/<healthCode>/<recordId>_countdown.m4a
to coexist with existing sustained phonation files.

Usage:
    python scripts/download_mpower_countdown.py --out-dir data/mpower
"""

import argparse
import csv
import logging
import math
import time
from pathlib import Path

import synapseclient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

VOICE_TABLE = "syn5511444"


def main():
    parser = argparse.ArgumentParser(description="Download mPower countdown audio")
    parser.add_argument("--out-dir", type=Path, default=Path("data/mpower"))
    parser.add_argument("--max-files", type=int, default=0,
                        help="Max files to download (0 = all)")
    args = parser.parse_args()

    audio_dir = args.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "countdown_manifest.csv"

    # Resume from existing manifest
    downloaded = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                downloaded.add(row["recordId"])
        logger.info("Resuming: %d countdown files already downloaded", len(downloaded))

    syn = synapseclient.Synapse()
    syn.login(silent=True)

    # Query countdown column - filter to non-null entries only
    logger.info("Querying voice activity table for countdown files...")
    results = syn.tableQuery(
        f'SELECT recordId, healthCode, "audio_countdown.m4a" '
        f'FROM {VOICE_TABLE} '
        f'WHERE "audio_countdown.m4a" IS NOT NULL'
    )

    df = results.asDataFrame()
    logger.info("Found %d recordings with countdown audio", len(df))

    # Filter out already downloaded
    to_download = df[~df["recordId"].isin(downloaded)]
    logger.info("Need to download %d (skipping %d already done)",
                len(to_download), len(df) - len(to_download))

    if len(to_download) == 0:
        logger.info("Nothing to download!")
        return

    n_downloaded = 0
    n_failed = 0

    with open(manifest_path, "a", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=["recordId", "healthCode", "path"])
        if not downloaded:
            writer.writeheader()

        for _, row in to_download.iterrows():
            record_id = row["recordId"]
            health_code = row["healthCode"]
            file_handle_id = row["audio_countdown.m4a"]

            # Skip NaN file handles
            if file_handle_id is None or (isinstance(file_handle_id, float) and math.isnan(file_handle_id)):
                n_failed += 1
                continue

            file_handle_id = str(int(file_handle_id))

            dest_dir = audio_dir / health_code
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{record_id}_countdown.m4a"

            try:
                # Download individual file handle from Synapse
                syn._downloadFileHandle(
                    fileHandleId=file_handle_id,
                    objectId=VOICE_TABLE,
                    objectType="TableEntity",
                    destination=str(dest_path),
                )

                writer.writerow({
                    "recordId": record_id,
                    "healthCode": health_code,
                    "path": str(dest_path.relative_to(args.out_dir)),
                })
                mf.flush()  # Flush after each write for resume safety

                n_downloaded += 1
                if n_downloaded % 100 == 0:
                    logger.info("Downloaded %d / %d files (%.1f%%)",
                                n_downloaded, len(to_download),
                                100 * n_downloaded / len(to_download))

            except Exception as e:
                n_failed += 1
                if n_failed <= 10:
                    logger.warning("Failed to download %s (fh=%s): %s",
                                   record_id, file_handle_id, e)
                elif n_failed == 11:
                    logger.warning("Suppressing further download warnings...")

                # Brief pause on failure to avoid hammering the API
                time.sleep(0.5)

            if args.max_files > 0 and n_downloaded >= args.max_files:
                logger.info("Reached max_files limit (%d)", args.max_files)
                break

    logger.info("Done: %d downloaded, %d skipped (already done), %d failed",
                n_downloaded, len(df) - len(to_download), n_failed)


if __name__ == "__main__":
    main()
