#!/usr/bin/env python3
"""Validate the v3 participants.parquet has the counts Alex agreed with us."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
PARQUET = PROJECT_ROOT / "data" / "processed_v3" / "participants.parquet"


def main():
    if not PARQUET.exists():
        sys.exit(f"Missing: {PARQUET}")

    p = pd.read_parquet(PARQUET)
    print(f"\n=== v3 cohort validation — {PARQUET} ===")
    print(f"Total participants: {len(p)}")
    assert len(p) == 984, f"Expected 984 total participants, got {len(p)}"

    # ---- cohort_split
    assert "cohort_split" in p.columns, "Missing cohort_split column"
    split = p["cohort_split"].value_counts()
    print(f"\nCohort split:\n{split.to_string()}")
    assert split.get("train", 0) == 846, f"Expected 846 train, got {split.get('train', 0)}"
    assert split.get("test", 0) == 138, f"Expected 138 test, got {split.get('test', 0)}"

    # ---- GSD control counts by split
    assert "gsd_control" in p.columns, "Missing gsd_control column"
    ctrl_by_split = p.groupby("cohort_split")["gsd_control"].sum()
    print(f"\ngsd_control by split:\n{ctrl_by_split.to_string()}")
    # Train control target 161 (post mutual-exclusivity drop of 2); window [155,170].
    # Test control target 38; window [34,42].
    train_ctrl = int(ctrl_by_split.get("train", 0))
    test_ctrl = int(ctrl_by_split.get("test", 0))
    assert 155 <= train_ctrl <= 170, f"Train controls out of expected range [155,170]: {train_ctrl}"
    assert 34 <= test_ctrl <= 42, f"Test controls out of expected range [34,42]: {test_ctrl}"

    # ---- Mutual exclusivity: no gsd_control=1 participant may have any disease flag
    disease_cols = [c for c in p.columns if c.startswith("gsd_") and c != "gsd_control"]
    overlap = p[(p["gsd_control"] == 1) & (p[disease_cols].sum(axis=1) > 0)]
    assert len(overlap) == 0, f"{len(overlap)} participants have gsd_control=1 AND a disease flag"

    # ---- No "Neither" participants (every row is either Control or Case)
    is_ctrl = p["gsd_control"] == 1
    has_disease = p[disease_cols].sum(axis=1) > 0
    neither = ~(is_ctrl | has_disease)
    assert neither.sum() == 0, f"{int(neither.sum())} 'Neither' participants survived preprocessing"

    # ---- disease category prevalence (sanity)
    print("\nDisease category counts:")
    for cat in ["cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]:
        assert cat in p.columns, f"Missing {cat}"
        n = int(p[cat].sum())
        print(f"  {cat}: {n}")
    # voice should be ~313, neuro ~225, mood ~~110, respiratory ~190 (very rough)
    assert p["cat_voice"].sum() > 200, "Voice case count implausibly low"
    assert p["cat_neuro"].sum() > 150, "Neuro case count implausibly low"
    assert p["cat_respiratory"].sum() > 150, "Respiratory case count implausibly low"

    # ---- Per-GSD flag spot-check
    print("\nGSD flag counts (top-level sanity):")
    for flag in ["gsd_parkinsons", "gsd_alz_dementia_mci", "gsd_airway_stenosis",
                 "gsd_copd_asthma", "gsd_depression", "gsd_bipolar", "gsd_anxiety"]:
        if flag in p.columns:
            print(f"  {flag}: {int(p[flag].sum())}")

    # ---- site column on recordings
    RECORDINGS_PATH = PROJECT_ROOT / "data" / "processed_v3" / "recordings.parquet"
    if RECORDINGS_PATH.exists():
        r = pd.read_parquet(RECORDINGS_PATH)
        kept_ids = set(p.index) if p.index.name == "record_id" else set(p["record_id"])
        r_kept = r[r["record_id"].isin(kept_ids)]
        n_kept = len(r_kept)
        n_site = int(r_kept["site"].notna().sum())
        print(f"\nRecordings for {len(kept_ids)} kept participants: {n_kept}")
        print(f"Recordings with site populated: {n_site}")
        # Allow up to 50 missing (Wasabi listing vs REDCap mismatch edge cases)
        missing = n_kept - n_site
        assert missing <= 50, f"Too many recordings missing site: {missing} of {n_kept}"
        print(f"Recordings missing site: {missing} (tolerance <= 50)")
        print("\nPer-site recording counts:")
        print(r_kept["site"].value_counts(dropna=False).to_string())

    print("\n[OK] All v3 cohort assertions passed.")


if __name__ == "__main__":
    main()
