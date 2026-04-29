#!/usr/bin/env python3
"""Table 1: Cohort demographics for VoiceFM paper.

Reads participants.parquet and site_mapping.csv to produce a cohort table
broken down by GSD disease category.

Usage:
    python scripts/paper_table1_cohort_v3.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def fmt_n_pct(n, total):
    if total == 0:
        return "—"
    return f"{n} ({100 * n / total:.1f}%)"


def fmt_mean_sd(series):
    valid = series[series >= 0]  # sentinel -1 = missing
    if len(valid) == 0:
        return "—"
    return f"{valid.mean():.1f} ± {valid.std():.1f}"


def build_table(pq, out_csv_name, table_label):
    """Build cohort table for a (filtered) participants dataframe."""

    # Define groups
    groups = {
        "Control": pq["gsd_control"] == 1,
        "Voice": pq["cat_voice"] == 1,
        "Neuro": pq["cat_neuro"] == 1,
        "Mood": pq["cat_mood"] == 1,
        "Respiratory": pq["cat_respiratory"] == 1,
        "Total": pd.Series(True, index=pq.index),
    }

    # Gender mapping
    gender_map = {0: "Female", 1: "Male", 2: "Non-binary", 3: "Other"}
    # Ethnicity mapping
    eth_map = {0: "Not Hispanic/Latino", 1: "Hispanic/Latino", 2: "Prefer not to answer"}
    # Education mapping
    edu_map = {0: "No formal/some HS", 1: "HS–Associates", 2: "College", 3: "Graduate+"}
    # Site display names (keys match site_mapping.csv values)
    site_display = {"USF": "USF", "Mt. Sinai": "Mt. Sinai", "VUMC": "VUMC", "WCM": "WCM", "MIT": "MIT"}

    rows = []

    def add_row(label, values):
        rows.append({"Characteristic": label, **{g: v for g, v in zip(groups.keys(), values)}})

    # N
    ns = [mask.sum() for mask in groups.values()]
    add_row("N", [str(n) for n in ns])

    # Age
    add_row("Age, mean ± SD", [fmt_mean_sd(pq.loc[mask, "age"]) for mask in groups.values()])

    # Gender
    for code, label in gender_map.items():
        vals = []
        for name, mask in groups.items():
            sub = pq.loc[mask]
            n = (sub["gender"] == code).sum()
            vals.append(fmt_n_pct(n, len(sub)))
        add_row(f"  {label}", vals)

    # Ethnicity: Hispanic/Latino
    vals = []
    for name, mask in groups.items():
        sub = pq.loc[mask]
        n = (sub["ethnicity"] == 1).sum()
        vals.append(fmt_n_pct(n, len(sub)))
    add_row("Hispanic/Latino", vals)

    # Race
    race_cols = {
        "race_white": "White",
        "race_black": "Black/African American",
        "race_asian": "Asian",
        "race_indigenous": "Indigenous",
        "race_pacific_islander": "Pacific Islander",
        "race_other": "Other race",
    }
    for col, label in race_cols.items():
        if col not in pq.columns:
            continue
        vals = []
        for name, mask in groups.items():
            sub = pq.loc[mask]
            n = (sub[col] == 1).sum()
            vals.append(fmt_n_pct(n, len(sub)))
        add_row(f"  {label}", vals)

    # Education
    for code, label in edu_map.items():
        vals = []
        for name, mask in groups.items():
            sub = pq.loc[mask]
            n = (sub["education"] == code).sum()
            vals.append(fmt_n_pct(n, len(sub)))
        add_row(f"  {label}", vals)

    # Functional status
    func_cols = {
        "func_hearing": "Hearing difficulty",
        "func_cognition": "Cognition difficulty",
        "func_mobility": "Mobility difficulty",
        "func_self_care": "Self-care difficulty",
        "func_independent_living": "Independent living difficulty",
    }
    for col, label in func_cols.items():
        if col not in pq.columns:
            continue
        vals = []
        for name, mask in groups.items():
            sub = pq.loc[mask]
            n = (sub[col] == 1).sum()
            vals.append(fmt_n_pct(n, len(sub)))
        add_row(f"  {label}", vals)

    # Smoking
    if "smoking_ever" in pq.columns:
        vals = []
        for name, mask in groups.items():
            sub = pq.loc[mask]
            n = (sub["smoking_ever"] == 1).sum()
            vals.append(fmt_n_pct(n, len(sub)))
        add_row("Ever smoked", vals)

    # Questionnaire scores
    for col, label in [("phq9_total", "PHQ-9"), ("gad7_total", "GAD-7"), ("vhi10_total", "VHI-10")]:
        if col in pq.columns:
            add_row(f"{label}, mean ± SD", [fmt_mean_sd(pq.loc[mask, col]) for mask in groups.values()])

    # Site
    if "site" in pq.columns:
        for site_key, site_label in site_display.items():
            vals = []
            for name, mask in groups.items():
                sub = pq.loc[mask]
                n = (sub["site"] == site_key).sum()
                vals.append(fmt_n_pct(n, len(sub)))
            add_row(f"  {site_label}", vals)

    # Recordings
    vals = []
    for name, mask in groups.items():
        sub = pq.loc[mask]
        vals.append(str(sub["n_recordings"].sum()))
    add_row("Recordings, N", vals)

    # Build DataFrame and display
    df = pd.DataFrame(rows)

    # Print formatted table
    print("\n" + "=" * 120)
    print("  Table 1: Cohort Demographics by GSD Disease Category")
    print("=" * 120)

    # Header
    header = f"{'Characteristic':<35}"
    for col in df.columns[1:]:
        header += f"{col:>14}"
    print(header)
    print("-" * 120)

    for _, row in df.iterrows():
        line = f"{row['Characteristic']:<35}"
        for col in df.columns[1:]:
            line += f"{row[col]:>14}"
        print(line)

    print("=" * 120)
    print("\nNote: Disease categories are not mutually exclusive; participants may appear in multiple columns.")
    print("PHQ-9, GAD-7, VHI-10: missing values (sentinel -1) excluded from mean ± SD.\n")

    # Save CSV
    out_path = PROJECT_ROOT / "paper_v3" / out_csv_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[{table_label}] Saved to {out_path}")
    return df


def main():
    pq = pd.read_parquet(PROJECT_ROOT / "data" / "processed_v3" / "participants.parquet")

    # Site mapping via recordings.parquet + site_mapping.csv
    rec_path = PROJECT_ROOT / "data" / "processed_v3" / "recordings.parquet"
    site_map_path = PROJECT_ROOT / "data" / "metadata" / "site_mapping.csv"
    if rec_path.exists() and site_map_path.exists():
        rec = pd.read_parquet(rec_path)
        sm = pd.read_csv(site_map_path)
        rec_with_site = rec[["record_id", "recording_id"]].merge(sm, on="recording_id", how="left")
        participant_site = rec_with_site.groupby("record_id")["site"].first()
        pq["site"] = pq.index.map(participant_site)
        rec_counts = rec.groupby("record_id").size()
        pq["n_recordings"] = pq.index.map(rec_counts).fillna(0).astype(int)
    else:
        pq["site"] = "Unknown"
        pq["n_recordings"] = 0

    # Full cohort
    build_table(pq, "table1_cohort_v3.csv", "Full cohort (n=984)")
    # Training split
    train_pq = pq[pq["cohort_split"] == "train"].copy() if "cohort_split" in pq.columns else pq
    build_table(train_pq, "table1a_cohort_train_v3.csv", f"Training cohort (n={len(train_pq)})")
    # Held-out test split
    test_pq = pq[pq["cohort_split"] == "test"].copy() if "cohort_split" in pq.columns else pq.iloc[:0]
    build_table(test_pq, "table1b_cohort_test_v3.csv", f"Validation cohort (n={len(test_pq)})")


if __name__ == "__main__":
    main()
