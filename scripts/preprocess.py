#!/usr/bin/env python3
"""Run the full preprocessing pipeline: clinical data + recording manifest."""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.clinical_encoder import ClinicalFeatureProcessor
from src.data.recording_manifest import RecordingManifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess B2AI clinical + recording data")
    parser.add_argument("--use-gsd", action="store_true", help="Use GSD diagnosis flags")
    parser.add_argument(
        "--redcap-csv", type=str,
        default=str(DATA_DIR / "metadata" / "bridge2ai_voice_redcap_data_v2.3.0_2026-02-01T00.00.00.304Z.csv"),
        help="Path to REDCap CSV export (default: v2.3.0)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(DATA_DIR / "processed_v3"),
        help="Output directory for participants.parquet and recordings.parquet",
    )
    parser.add_argument(
        "--v23-csv", type=str, default=None,
        help="Path to v2.3.0 REDCap (adds cohort_split=train/test column based on record_id membership). "
             "Only used when --redcap-csv points at a newer export.",
    )
    args = parser.parse_args()

    redcap_csv = Path(args.redcap_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process clinical data
    logger.info("Processing clinical data...")
    processor = ClinicalFeatureProcessor(use_gsd=args.use_gsd)
    participants = processor.process(str(redcap_csv), v23_csv_path=args.v23_csv)
    participants.to_parquet(output_dir / "participants.parquet")
    logger.info(f"Saved {len(participants)} participants to participants.parquet")

    feature_config = processor.get_feature_names()
    logger.info(f"Features: {len(feature_config['binary'])} binary, "
                f"{len(feature_config['continuous'])} continuous, "
                f"{len(feature_config.get('categorical', {}))} categorical")

    # 2. Build recording manifest
    logger.info("Building recording manifest...")
    manifest_builder = RecordingManifest()
    recordings = manifest_builder.process(str(redcap_csv))
    recordings.to_parquet(output_dir / "recordings.parquet")
    logger.info(f"Saved {len(recordings)} recordings to recordings.parquet")

    # 3. Summary
    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info(f"  Participants: {len(participants)}")
    logger.info(f"  Recordings: {len(recordings)}")
    logger.info(f"  Unique task types: {recordings['recording_name'].nunique()}")
    logger.info(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
