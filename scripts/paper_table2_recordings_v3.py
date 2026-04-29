#!/usr/bin/env python3
"""Table 2: Recording details by task category.

Usage:
    python scripts/paper_table2_recordings_v3.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent


def categorize_recording(name):
    """Map recording name to task category."""
    nl = name.lower()
    if "audio check" in nl:
        return "Audio Check"
    elif "cape v" in nl:
        return "CAPE-V Sentences"
    elif "harvard" in nl:
        return "Harvard Sentences"
    elif "diadoch" in nl:
        return "Diadochokinesis"
    elif "caterpillar" in nl or "rainbow" in nl:
        return "Reading Passages"
    elif "cinderella" in nl or "picture" in nl or "free speech" in nl or "open response" in nl or "open_response" in nl or "story" in nl:
        return "Free Speech / Narrative"
    elif "breath" in nl or "cough" in nl or "respiration" in nl:
        return "Respiration / Cough"
    elif "glide" in nl or "vowel" in nl or "prolonged" in nl or "maximum" in nl or "mpt" in nl:
        return "Sustained Phonation"
    elif "loudness" in nl:
        return "Loudness Tasks"
    elif "fluency" in nl:
        return "Verbal Fluency"
    elif "word" in nl or "confrontation" in nl:
        return "Confrontation Naming"
    elif "counting" in nl:
        return "Counting"
    elif "productive vocabulary" in nl:
        return "Productive Vocabulary"
    elif "random item" in nl:
        return "Random Item Generation"
    else:
        return "Other"


# Desired display order
CATEGORY_ORDER = [
    "Free Speech / Narrative",
    "Respiration / Cough",
    "Sustained Phonation",
    "Diadochokinesis",
    "Harvard Sentences",
    "CAPE-V Sentences",
    "Productive Vocabulary",
    "Reading Passages",
    "Loudness Tasks",
    "Confrontation Naming",
    "Verbal Fluency",
    "Random Item Generation",
    "Counting",
    "Audio Check",
    "Other",
]


def main():
    rec = pd.read_parquet(PROJECT_ROOT / "data" / "processed_v3" / "recordings.parquet")
    parts = pd.read_parquet(PROJECT_ROOT / "data" / "processed_v3" / "participants.parquet")
    # Restrict to cohort participants (excludes "Neither" / dropped IDs that have recordings but no cohort label)
    rec = rec[rec["record_id"].isin(parts.index)].copy()

    # Load site mapping
    site_map = pd.read_csv(PROJECT_ROOT / "data" / "metadata" / "site_mapping.csv")
    rec = rec[["record_id", "recording_id", "recording_name", "recording_duration"]].merge(
        site_map[["recording_id", "site"]], on="recording_id", how="left"
    )

    rec["category"] = rec["recording_name"].apply(categorize_recording)

    rows = []
    for cat in CATEGORY_ORDER:
        sub = rec[rec["category"] == cat]
        if len(sub) == 0:
            continue
        rows.append({
            "Task Category": cat,
            "Recordings": len(sub),
            "Participants": sub["record_id"].nunique(),
            "Types": sub["recording_name"].nunique(),
            "Mean Duration (s)": f"{sub['recording_duration'].mean():.1f}",
            "Total Duration (h)": f"{sub['recording_duration'].sum() / 3600:.1f}",
        })

    # Add total row
    rows.append({
        "Task Category": "Total",
        "Recordings": len(rec),
        "Participants": rec["record_id"].nunique(),
        "Types": rec["recording_name"].nunique(),
        "Mean Duration (s)": f"{rec['recording_duration'].mean():.1f}",
        "Total Duration (h)": f"{rec['recording_duration'].sum() / 3600:.1f}",
    })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("  Table 2: Recording Details by Task Category")
    print("=" * 100)
    print(f"\n  {'Task Category':<28} {'Recordings':>12} {'Participants':>14} {'Types':>8} {'Mean Dur (s)':>14} {'Total (h)':>12}")
    print(f"  {'-' * 90}")
    for _, row in df.iterrows():
        print(f"  {row['Task Category']:<28} {row['Recordings']:>12} {row['Participants']:>14} {row['Types']:>8} {row['Mean Duration (s)']:>14} {row['Total Duration (h)']:>12}")
    print("=" * 100)

    out_dir = PROJECT_ROOT / "paper_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "table2_recordings_v3.csv", index=False)
    print(f"\nSaved to {out_dir / 'table2_recordings_v3.csv'}")


if __name__ == "__main__":
    main()
