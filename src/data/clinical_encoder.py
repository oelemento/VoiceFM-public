"""
Clinical data preprocessing for VoiceFM.

Loads the Bridge2AI REDCap CSV export and produces a clean per-participant
feature table (participants.parquet) with demographics, condition flags,
questionnaire scores, and derived disease-category labels.

Supports two label modes:
- use_gsd=False (default): self-reported condition flags from enrollment checkboxes
- use_gsd=True: Gold Standard Diagnosis (GSD) from clinician-validated diagnosis forms
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Self-reported condition flags (original) ──────────────────────────────

VOICE_FLAGS = [
    "glottic_insufficiency",
    "laryng_cancer",
    "laryngitis",
    "benign_cord_lesion",
    "mtd",
    "rrp",
    "spas_dys",
    "voc_fold_paralysis",
]

NEURO_FLAGS = [
    "alz_dementia_mci",
    "als",
    "huntingtons",
    "parkinsons",
]

MOOD_FLAGS = [
    "alcohol_subst_abuse",
    "gad",
    "add_adhd",
    "asd",
    "bipolar",
    "bpd",
    "depression",
    "ed",
    "insomnia",
    "ocd",
    "panic",
    "ptsd",
    "schizophrenia",
    "soc_anx_dis",
    "other_psych",
]

RESPIRATORY_FLAGS = [
    "asthma",
    "airway_stenosis",
    "chronic_cough",
    "copd",
    "osa",
]

ALL_CONDITION_FLAGS = VOICE_FLAGS + NEURO_FLAGS + MOOD_FLAGS + RESPIRATORY_FLAGS

# ── GSD condition flags (clinician-validated) ─────────────────────────────

# Maps flag name -> primary GSD confirmation column in the REDCap CSV.
# Each diagnosis form has a primary column that confirms the clinician's diagnosis.
# We use these specific columns instead of "any non-null in form" to avoid
# false positives from screening/administrative fields.
GSD_PRIMARY_COL = {
    "gsd_control": "diagnosis_c_ac",  # control = is_control_participant=="Yes" AND diagnosis_c_ac=="No"
    "gsd_glottic_insufficiency": "diagnois_gi_gsd",  # REDCap typo is intentional
    "gsd_benign_lesion": "diagnois_bl_gsd",  # REDCap typo is intentional
    "gsd_laryng_cancer": "diagnosis_lc_gsd",
    "gsd_laryngitis": "diagnosis_l_gsd",
    "gsd_laryngeal_dystonia": "diagnosis_ld_gsd",
    "gsd_mtd": "diagnosis_mtd_gsd",
    "gsd_precancerous_lesion": "diagnosis_pl_gsd",
    "gsd_vocal_fold_paralysis": "diagnosis_vfp_gsd",
    "gsd_parkinsons": "diagnosis_parkinsons_gsd",
    "gsd_huntingtons": "diagnosis_hd_gsd",
    "gsd_als": "diagnosis_als_gsd",
    "gsd_alz_dementia_mci": "diagnosis_alz_dementia_mci_gsd",
    "gsd_anxiety": "diagnosis_diagnosed_ad",
    "gsd_depression": "dmdd_diagnosed_dd",
    "gsd_bipolar": "mbd_diagnosed_bm",
    "gsd_airway_stenosis": "diagnosis_as_gsd",
    "gsd_copd_asthma": "diagnosis_ca_copd_asthma",
    "gsd_chronic_cough": "diagnosis_ucc_persisted",
}

GSD_VOICE_FLAGS = [
    "gsd_glottic_insufficiency", "gsd_benign_lesion", "gsd_laryng_cancer",
    "gsd_laryngitis", "gsd_laryngeal_dystonia", "gsd_mtd",
    "gsd_precancerous_lesion", "gsd_vocal_fold_paralysis",
]
GSD_NEURO_FLAGS = [
    "gsd_parkinsons", "gsd_huntingtons", "gsd_als", "gsd_alz_dementia_mci",
]
GSD_MOOD_FLAGS = [
    "gsd_anxiety", "gsd_depression", "gsd_bipolar",
]
GSD_RESPIRATORY_FLAGS = [
    "gsd_airway_stenosis", "gsd_copd_asthma", "gsd_chronic_cough",
]
ALL_GSD_FLAGS = (
    ["gsd_control"] + GSD_VOICE_FLAGS + GSD_NEURO_FLAGS
    + GSD_MOOD_FLAGS + GSD_RESPIRATORY_FLAGS
)

# Values that count as positive for the combined COPD/Asthma diagnosis field.
# Per Alex's spec the field value is "COPD only", "Asthma only", or
# "Both COPD and asthma" (instead of plain "Yes").
COPD_ASTHMA_VALUES = frozenset({"COPD only", "Asthma only", "Both COPD and asthma"})

# ── Shared constants ─────────────────────────────────────────────────────

DISEASE_CATEGORIES = ["cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]

RACE_FLAGS = [
    "race_indigenous",  # American Indian or Alaska Native + Canadian Indigenous
    "race_asian",
    "race_black",
    "race_pacific_islander",
    "race_white",
    "race_other",
]

FUNCTIONAL_FLAGS = [
    "func_hearing", "func_cognition", "func_mobility",
    "func_self_care", "func_independent_living",
]

SMOKING_FLAGS = ["smoking_ever", "smoking_missing"]

# PHQ-9: standard 0-3 Likert
PHQ9_ITEMS = [
    "no_interest",
    "feeling_depressed",
    "trouble_sleeping",
    "no_energy",
    "no_appetite",
    "feeling_bad_self",
    "trouble_concentrate",
    "move_speak_slow",
    "thoughts_death",
]

PHQ9_MAP = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3,
}

# GAD-7: same 0-3 Likert
GAD7_ITEMS = [
    "nervous_anxious",
    "cant_control_worry",
    "worry_too_much",
    "trouble_relaxing",
    "hard_to_sit_still",
    "easily_agitated",
    "afraid_of_things",
]

GAD7_MAP = PHQ9_MAP  # identical scale labels

# VHI-10: 0-4 frequency Likert
VHI10_ITEMS = [
    "voice_difficult_hear",
    "tough_to_understand",
    "voice_restrict_social",
    "left_out_convo",
    "voice_lose_income",
    "strain_voice",
    "voice_clarity",
    "voice_upsetting",
    "voice_handicapped",
    "ask_whats_wrong_voice",
]

VHI10_MAP = {
    "Never": 0,
    "Almost Never": 1,
    "Sometimes": 2,
    "Almost Always": 3,
    "Always": 4,
}


# ── helpers ─────────────────────────────────────────────────────────────────

def _encode_binary(series: pd.Series) -> pd.Series:
    """Checked -> 1, everything else (Unchecked / NaN) -> 0."""
    return (series == "Checked").astype(np.int8)


def _encode_control(series: pd.Series) -> pd.Series:
    """Yes -> 1, No -> 0, NaN -> 0."""
    return (series == "Yes").astype(np.int8)


def _parse_age(series: pd.Series) -> pd.Series:
    """Convert age column to float; '90 and above' -> 90.0, NaN -> median."""
    cleaned = series.replace("90 and above", "90.0")
    numeric = pd.to_numeric(cleaned, errors="coerce")
    median_age = numeric.median()
    return numeric.fillna(median_age).astype(np.float32)


def _encode_language(series: pd.Series) -> pd.Series:
    """Map language strings to integer codes. NaN -> 0."""
    uniq = sorted(series.dropna().unique())
    mapping = {lang: i for i, lang in enumerate(uniq)}
    return series.map(mapping).fillna(0).astype(np.int16)


def _encode_categorical(series: pd.Series, ordered_values: list[str]) -> pd.Series:
    """Map string values to integer codes. Unknown/NaN -> 0."""
    mapping = {val: i for i, val in enumerate(ordered_values)}
    return series.map(mapping).fillna(0).astype(np.int16)


def _score_questionnaire(
    df_full: pd.DataFrame,
    instrument_name: str,
    items: list[str],
    likert_map: dict[str, int],
    precomputed_col: str | None = None,
) -> pd.Series:
    """
    Extract a single questionnaire total score per participant.

    If a pre-computed score column exists and is populated for a row, use it.
    Otherwise sum the item-level Likert values.
    For participants with multiple instances, keep the highest repeat_instance.

    Returns a Series indexed by record_id with float scores (NaN if missing).
    """
    q_rows = df_full[df_full["redcap_repeat_instrument"] == instrument_name].copy()
    if q_rows.empty:
        logger.warning("No rows found for instrument '%s'", instrument_name)
        return pd.Series(dtype=np.float32, name="score")

    # Sort so highest repeat_instance comes last; we keep last per participant.
    q_rows = q_rows.sort_values("redcap_repeat_instance")

    # Try precomputed score first
    has_precomputed = (
        precomputed_col is not None
        and precomputed_col in q_rows.columns
    )

    # Compute item-level sum for every row regardless (fallback)
    for item in items:
        q_rows[f"__{item}_num"] = q_rows[item].map(likert_map)
    numeric_items = [f"__{item}_num" for item in items]
    q_rows["_item_sum"] = q_rows[numeric_items].sum(axis=1, min_count=1)

    if has_precomputed:
        # Use precomputed where available, item-sum otherwise
        q_rows["_score"] = q_rows[precomputed_col].combine_first(q_rows["_item_sum"])
    else:
        q_rows["_score"] = q_rows["_item_sum"]

    # Keep last (highest repeat_instance) per participant
    scores = (
        q_rows.drop_duplicates(subset="record_id", keep="last")
        .set_index("record_id")["_score"]
        .astype(np.float32)
    )
    return scores


def _extract_gsd_flags(
    df: pd.DataFrame,
    participant_ids: pd.Index,
) -> pd.DataFrame:
    """Extract GSD condition flags per Alex Sigaras's authoritative spec.

    Control: participants for whom ``is_control_participant == "Yes"`` AND
    ``diagnosis_c_ac == "No"`` (adjudicator confirmed no condition affecting
    speech/voice).

    Disease flags: participants for whom the primary diagnosis column equals
    ``"Yes"``.  Exception: the combined COPD/Asthma field uses categorical
    values (``COPD only`` / ``Asthma only`` / ``Both COPD and asthma``).

    "No" and "Not certain" are treated as NOT positive for that condition.

    Parameters
    ----------
    df : Full REDCap DataFrame (all rows including repeats).
    participant_ids : Index of record_ids to produce flags for.

    Returns
    -------
    DataFrame indexed by record_id with binary GSD flag columns.
    """

    def _first_non_null(s: pd.Series):
        s = s.dropna()
        return s.iloc[0] if len(s) > 0 else None

    result = pd.DataFrame(0, index=participant_ids, columns=ALL_GSD_FLAGS, dtype=np.int8)

    # ── Control: is_control_participant=="Yes" AND diagnosis_c_ac=="No" ────
    ctrl_cols = ["is_control_participant", "diagnosis_c_ac"]
    missing = [c for c in ctrl_cols if c not in df.columns]
    if missing:
        logger.warning("GSD control: missing columns %s — control flag will be all zero", missing)
    else:
        ctrl_raw = df.groupby("record_id")[ctrl_cols].agg(_first_non_null)
        ctrl_match = (ctrl_raw["is_control_participant"] == "Yes") & (ctrl_raw["diagnosis_c_ac"] == "No")
        ctrl_ids = ctrl_raw.index[ctrl_match]
        matched = participant_ids.intersection(ctrl_ids)
        result.loc[matched, "gsd_control"] = 1
        logger.info("GSD gsd_control (is_control=Yes AND diagnosis_c_ac=No): %d participants", len(matched))

    # ── Disease flags: primary column == "Yes" (or COPD/Asthma values) ────
    for flag_name, primary_col in GSD_PRIMARY_COL.items():
        if flag_name == "gsd_control":
            continue  # handled above

        if primary_col not in df.columns:
            logger.warning(
                "GSD primary column '%s' for %s not found in CSV", primary_col, flag_name,
            )
            continue

        col_per_pid = df.groupby("record_id")[primary_col].agg(_first_non_null)

        if flag_name == "gsd_copd_asthma":
            match = col_per_pid.isin(COPD_ASTHMA_VALUES)
        else:
            match = col_per_pid == "Yes"

        matched_ids = participant_ids.intersection(col_per_pid.index[match])
        result.loc[matched_ids, flag_name] = 1
        logger.info("GSD %s (%s): %d participants", flag_name, primary_col, len(matched_ids))

    # Mutual exclusivity: if any disease flag is 1, force gsd_control to 0.
    # A participant with a confirmed disease diagnosis is a Case, not a Control,
    # even if they self-reported as a Control at enrollment.
    disease_cols = [c for c in ALL_GSD_FLAGS if c != "gsd_control"]
    has_disease = result[disease_cols].sum(axis=1) > 0
    flipped = (result["gsd_control"] == 1) & has_disease
    if flipped.any():
        result.loc[flipped, "gsd_control"] = 0
        logger.info(
            "Mutual exclusivity: cleared gsd_control on %d participants who also had a disease Yes",
            int(flipped.sum()),
        )
        if int(flipped.sum()) > 10:
            logger.warning(
                "Unusually large mutual-exclusivity flip (%d participants) — check GSD fields",
                int(flipped.sum()),
            )

    return result


def _extract_demographics(
    df: pd.DataFrame,
    participant_ids: pd.Index,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Extract demographic features from the demographics questionnaire.

    Returns
    -------
    (features_df, categorical_sizes) where features_df is indexed by record_id
    and categorical_sizes maps categorical column names to their vocabulary size.
    """
    demo = df[df["redcap_repeat_instrument"] == "Q - Generic - Demographics"].copy()
    # Keep last instance per participant
    demo = demo.sort_values("redcap_repeat_instance")
    demo = demo.drop_duplicates(subset="record_id", keep="last").set_index("record_id")
    demo = demo.reindex(participant_ids)

    features = pd.DataFrame(index=participant_ids)

    # Gender: 4 categories (female=0, male=1, nonBinary=2, other/unknown=3)
    gender_values = [
        "Female gender identity",
        "Male gender identity",
        "Non-binary or genderqueer gender identity",
        "Other",  # Also catches "Prefer not to answer" -> 0 via fillna
    ]
    features["gender"] = _encode_categorical(demo["gender_identity"], gender_values)

    # Ethnicity: 3 categories (not_hispanic=0, hispanic=1, unknown=2)
    ethnicity_values = [
        "Not Hispanic or Latino",
        "Hispanic or Latino",
        "Prefer not to answer",
    ]
    features["ethnicity"] = _encode_categorical(demo["ethnicity"], ethnicity_values)

    # Education: binned to 4 ordinal levels
    edu_mapping = {
        "No formal education": 0,
        "Some elementary school": 0,
        "Some secondary or high school education": 0,
        "High School or secondary school degree complete": 1,
        "Some college education": 1,
        "Associate's or technical degree complete": 1,
        "College or baccalaureate degree complete": 2,
        "Some post-baccalaureate education": 2,
        "Graduate or professional degree complete": 3,
        "Doctoral or post graduate education": 3,
        "Other": 0,
        "Prefer not to answer": 0,
    }
    features["education"] = demo["edu_level"].map(edu_mapping).fillna(0).astype(np.int16)

    # Race: multi-binary (6 flags)
    # race___1 = American Indian/Alaska Native, race___2 = Asian,
    # race___3 = Black/African American, race___4 = Native Hawaiian/Pacific Islander,
    # race___5 = White, race___6 = Canadian Indigenous, race___7 = Other,
    # race___8 = Prefer not to answer
    race_col_map = {
        "race_indigenous": ["race___1", "race___6"],  # American Indian + Canadian Indigenous
        "race_asian": ["race___2"],
        "race_black": ["race___3"],
        "race_pacific_islander": ["race___4"],
        "race_white": ["race___5"],
        "race_other": ["race___7", "race___8"],  # Other + Prefer not to answer
    }
    for flag, cols in race_col_map.items():
        existing = [c for c in cols if c in demo.columns]
        if existing:
            # REDCap multi-select: NaN = not selected, non-NaN = selected (stores label text)
            features[flag] = demo[existing].notna().any(axis=1).astype(np.int8)
        else:
            features[flag] = np.int8(0)

    # Functional indicators: binary (Yes=1, else=0)
    func_col_map = {
        "func_hearing": "hearing",
        "func_cognition": "cognition",
        "func_mobility": "mobility",
        "func_self_care": "self_care",
        "func_independent_living": "independent_living",
    }
    for flag, col in func_col_map.items():
        if col in demo.columns:
            features[flag] = (demo[col] == "Yes").astype(np.int8)
        else:
            features[flag] = np.int8(0)

    # Smoking: from confounders questionnaire
    conf = df[df["redcap_repeat_instrument"] == "Q - Generic - Confounders"].copy()
    conf = conf.sort_values("redcap_repeat_instance")
    conf = conf.drop_duplicates(subset="record_id", keep="last").set_index("record_id")
    conf = conf.reindex(participant_ids)

    if "smoking_entire_life" in conf.columns:
        features["smoking_ever"] = (conf["smoking_entire_life"] == "Yes").astype(np.int8)
        features["smoking_missing"] = conf["smoking_entire_life"].isna().astype(np.int8)
    else:
        features["smoking_ever"] = np.int8(0)
        features["smoking_missing"] = np.int8(1)

    categorical_sizes = {
        "gender": 4,
        "ethnicity": 3,
        "education": 4,
    }

    return features, categorical_sizes


# ── main class ──────────────────────────────────────────────────────────────

class ClinicalFeatureProcessor:
    """Preprocesses the REDCap CSV into a clean per-participant feature table.

    Parameters
    ----------
    use_gsd : if True, feature lists use GSD flags; if False, self-reported flags.
        This sets the correct defaults for get_feature_names() even without
        calling process(). Training scripts create a ClinicalFeatureProcessor
        just to get feature_config, so this must be correct from __init__.
    """

    _continuous_cols: list[str] = ["age", "phq9_total", "gad7_total", "vhi10_total"]

    def __init__(self, use_gsd: bool = False):
        self._use_gsd = use_gsd
        if use_gsd:
            condition_flags = ALL_GSD_FLAGS
        else:
            condition_flags = ["is_control_participant"] + ALL_CONDITION_FLAGS
        self._binary_cols = (
            list(condition_flags)
            + list(DISEASE_CATEGORIES)
            + list(RACE_FLAGS)
            + list(FUNCTIONAL_FLAGS)
            + list(SMOKING_FLAGS)
        )
        self._categorical_cols = ["selected_language", "gender", "ethnicity", "education"]
        # Default sizes; process() updates with actual data
        self._categorical_sizes = {
            "selected_language": 2,
            "gender": 4,
            "ethnicity": 3,
            "education": 4,
        }

    def process(
        self,
        csv_path: str | Path,
        use_gsd: bool | None = None,
        v23_csv_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Run the full clinical preprocessing pipeline.

        Parameters
        ----------
        csv_path : path to the Bridge2AI REDCap CSV export.
        use_gsd : if True, use GSD diagnosis forms instead of self-reported flags.
            Defaults to the value set in __init__.
        v23_csv_path : optional path to the v2.3.0 REDCap CSV. If provided, a
            ``cohort_split`` column is added tagging each participant as
            ``"train"`` (record_id present in v2.3.0) or ``"test"``
            (prospective test participant not in v2.3.0).

        Returns
        -------
        pd.DataFrame with one row per participant (record_id as index).
        """
        csv_path = Path(csv_path)
        if use_gsd is not None:
            self._use_gsd = use_gsd
        use_gsd = self._use_gsd
        logger.info("Loading REDCap CSV: %s", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info("Loaded %d rows x %d columns", *df.shape)

        # ── 1. Base participant rows ────────────────────────────────────
        base = df[df["redcap_repeat_instrument"].isna()].copy()
        logger.info("Base participant rows: %d", len(base))

        participants = base[["record_id"]].copy()
        participants = participants.set_index("record_id")

        # Demographics (from base row)
        participants["age"] = _parse_age(base.set_index("record_id")["age"])
        participants["selected_language"] = _encode_language(
            base.set_index("record_id")["selected_language"]
        )

        # ── 2. Condition flags + control status ─────────────────────────
        if use_gsd:
            logger.info("Using GSD (clinician-validated) condition flags")

            gsd_flags = _extract_gsd_flags(df, participants.index)
            for col in ALL_GSD_FLAGS:
                participants[col] = gsd_flags[col]

            # Disease category rollups from GSD flags
            participants["cat_voice"] = participants[GSD_VOICE_FLAGS].max(axis=1).astype(np.int8)
            participants["cat_neuro"] = participants[GSD_NEURO_FLAGS].max(axis=1).astype(np.int8)
            participants["cat_mood"] = participants[GSD_MOOD_FLAGS].max(axis=1).astype(np.int8)
            participants["cat_respiratory"] = participants[GSD_RESPIRATORY_FLAGS].max(axis=1).astype(np.int8)

            # Also include self-reported flags so the same parquet works for
            # cross-eval (self-reported model evaluated on GSD labels)
            participants["is_control_participant"] = _encode_control(
                base.set_index("record_id")["is_control_participant"]
            )
            for col in ALL_CONDITION_FLAGS:
                participants[col] = _encode_binary(base.set_index("record_id")[col])

            condition_flags = ALL_GSD_FLAGS
        else:
            logger.info("Using self-reported condition flags")

            # Control status
            participants["is_control_participant"] = _encode_control(
                base.set_index("record_id")["is_control_participant"]
            )
            # Condition flags
            for col in ALL_CONDITION_FLAGS:
                participants[col] = _encode_binary(base.set_index("record_id")[col])

            # Disease category rollups
            participants["cat_voice"] = participants[VOICE_FLAGS].max(axis=1).astype(np.int8)
            participants["cat_neuro"] = participants[NEURO_FLAGS].max(axis=1).astype(np.int8)
            participants["cat_mood"] = participants[MOOD_FLAGS].max(axis=1).astype(np.int8)
            participants["cat_respiratory"] = participants[RESPIRATORY_FLAGS].max(axis=1).astype(np.int8)

            condition_flags = ["is_control_participant"] + ALL_CONDITION_FLAGS

        # ── 3. Enriched demographics ────────────────────────────────────
        demo_features, demo_cat_sizes = _extract_demographics(df, participants.index)
        for col in demo_features.columns:
            participants[col] = demo_features[col]

        # ── 4. Questionnaire scores ────────────────────────────────────
        phq9_scores = _score_questionnaire(
            df,
            instrument_name="Q - Generic - PHQ-9",
            items=PHQ9_ITEMS,
            likert_map=PHQ9_MAP,
            precomputed_col=None,
        )
        gad7_scores = _score_questionnaire(
            df,
            instrument_name="Q - Generic - GAD-7 Anxiety",
            items=GAD7_ITEMS,
            likert_map=GAD7_MAP,
            precomputed_col=None,
        )
        vhi10_scores = _score_questionnaire(
            df,
            instrument_name="Q - Generic - VHI-10",
            items=VHI10_ITEMS,
            likert_map=VHI10_MAP,
            precomputed_col="vhi_10_calc_score",
        )

        participants["phq9_total"] = phq9_scores.reindex(participants.index)
        participants["gad7_total"] = gad7_scores.reindex(participants.index)
        participants["vhi10_total"] = vhi10_scores.reindex(participants.index)

        # Sentinel -1 for missing questionnaire scores
        for col in ["phq9_total", "gad7_total", "vhi10_total"]:
            participants[col] = participants[col].fillna(-1).astype(np.float32)

        # ── 5. Set feature column lists ────────────────────────────────
        self._binary_cols = (
            list(condition_flags)
            + list(DISEASE_CATEGORIES)
            + list(RACE_FLAGS)
            + list(FUNCTIONAL_FLAGS)
            + list(SMOKING_FLAGS)
        )
        self._categorical_cols = ["selected_language", "gender", "ethnicity", "education"]
        self._categorical_sizes = {
            "selected_language": int(participants["selected_language"].max()) + 1,
            **demo_cat_sizes,
        }

        # ── 6. Cohort split (prospective test identification) ───────────
        if v23_csv_path is not None:
            v23_path = Path(v23_csv_path)
            logger.info("Loading v2.3.0 REDCap for cohort split: %s", v23_path)
            v23_ids = set(pd.read_csv(v23_path, low_memory=False, usecols=["record_id"])["record_id"].unique())
            participants["cohort_split"] = participants.index.map(
                lambda rid: "train" if rid in v23_ids else "test"
            )
            n_train = (participants["cohort_split"] == "train").sum()
            n_test = (participants["cohort_split"] == "test").sum()
            logger.info("Cohort split: %d train, %d test", n_train, n_test)

        # ── 7. Drop "Neither" participants (not Control AND no disease) ──
        if use_gsd:
            disease_cols = [c for c in ALL_GSD_FLAGS if c != "gsd_control"]
            is_ctrl = participants["gsd_control"] == 1
            has_disease = participants[disease_cols].sum(axis=1) > 0
            keep = is_ctrl | has_disease
            dropped = int((~keep).sum())
            if dropped > 0:
                if "cohort_split" in participants.columns:
                    split_breakdown = participants.loc[~keep, "cohort_split"].value_counts().to_dict()
                    logger.info(
                        "Dropped %d 'Neither' participants (no Control AND no disease): %s",
                        dropped, split_breakdown,
                    )
                else:
                    logger.info("Dropped %d 'Neither' participants (no Control AND no disease)", dropped)
                if dropped > 50:
                    logger.warning(
                        "Unusually large 'Neither' drop (%d participants) — check REDCap column availability",
                        dropped,
                    )
                participants = participants[keep].copy()

        logger.info("Final participant table: %d rows x %d columns", *participants.shape)
        logger.info(
            "Features: %d binary, %d continuous, %d categorical",
            len(self._binary_cols), len(self._continuous_cols), len(self._categorical_sizes),
        )
        return participants

    def get_feature_names(self) -> dict:
        """Return feature column names grouped by type.

        Returns dict with:
            binary: list of column names
            continuous: list of column names
            categorical: dict mapping column name -> num_categories
        """
        cat_sizes = self._categorical_sizes or {name: 2 for name in self._categorical_cols}
        return {
            "binary": list(self._binary_cols),
            "continuous": list(self._continuous_cols),
            "categorical": cat_sizes,
        }

    def get_disease_categories(self) -> list[str]:
        """Return the 4 disease-category column names."""
        return list(DISEASE_CATEGORIES)


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    ROOT = Path(__file__).resolve().parents[2]  # VoiceFM/

    parser = argparse.ArgumentParser(description="Preprocess clinical features")
    parser.add_argument(
        "--use-gsd", action="store_true",
        help="Use GSD (clinician-validated) diagnosis flags instead of self-reported",
    )
    parser.add_argument(
        "--csv", type=str,
        default=str(ROOT / "data" / "metadata" / "bridge2ai_voice_redcap_data_v2.3.0_2026-02-01T00.00.00.304Z.csv"),
        help="Path to REDCap CSV export",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(ROOT / "data" / "processed" / "participants.parquet"),
        help="Output parquet path",
    )
    parser.add_argument(
        "--v23-csv", type=str, default=None,
        help="Path to v2.3.0 REDCap CSV (to compute cohort_split column for prospective test separation). "
             "If omitted, no cohort_split column is added.",
    )
    args = parser.parse_args()

    CSV_PATH = Path(args.csv)
    OUT_PATH = Path(args.output)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    processor = ClinicalFeatureProcessor(use_gsd=args.use_gsd)
    participants = processor.process(CSV_PATH, v23_csv_path=args.v23_csv)

    # Save
    participants.to_parquet(OUT_PATH, engine="pyarrow")
    logger.info("Saved to %s", OUT_PATH)

    # ── Summary stats ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PARTICIPANT TABLE SUMMARY")
    print("=" * 60)
    print(f"Shape: {participants.shape}")
    print(f"Label mode: {'GSD (clinician-validated)' if args.use_gsd else 'Self-reported'}")
    print()

    print("-- Demographics --")
    print(f"  Age:  mean={participants['age'].mean():.1f}  "
          f"median={participants['age'].median():.1f}  "
          f"min={participants['age'].min():.0f}  max={participants['age'].max():.0f}")
    print(f"  Languages: {participants['selected_language'].value_counts().to_dict()}")
    print()

    # Control status
    control_col = "gsd_control" if args.use_gsd else "is_control_participant"
    if control_col in participants.columns:
        print(f"  Controls ({control_col}): {participants[control_col].sum()} / {len(participants)}")
    print()

    print("-- Disease category prevalence --")
    for cat in DISEASE_CATEGORIES:
        n = participants[cat].sum()
        print(f"  {cat}: {n} ({100 * n / len(participants):.1f}%)")
    print()

    # Condition flags
    if args.use_gsd:
        flag_list = ALL_GSD_FLAGS
        print("-- GSD condition flag prevalence --")
    else:
        flag_list = ALL_CONDITION_FLAGS
        print("-- Self-reported condition flag prevalence (top 10) --")
    flag_counts = {f: int(participants[f].sum()) for f in flag_list}
    for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {flag}: {count}")
    print()

    print("-- New demographic features --")
    for col in ["gender", "ethnicity", "education"]:
        if col in participants.columns:
            print(f"  {col}: {participants[col].value_counts().sort_index().to_dict()}")
    for col in RACE_FLAGS:
        if col in participants.columns:
            print(f"  {col}: {int(participants[col].sum())} positive")
    for col in FUNCTIONAL_FLAGS:
        if col in participants.columns:
            print(f"  {col}: {int(participants[col].sum())} positive")
    for col in SMOKING_FLAGS:
        if col in participants.columns:
            print(f"  {col}: {int(participants[col].sum())}")
    print()

    print("-- Questionnaire scores (excluding missing=-1) --")
    for col in ["phq9_total", "gad7_total", "vhi10_total"]:
        valid = participants[col][participants[col] >= 0]
        if len(valid):
            print(f"  {col}: n={len(valid)}  "
                  f"mean={valid.mean():.1f}  median={valid.median():.1f}  "
                  f"min={valid.min():.0f}  max={valid.max():.0f}")
        else:
            print(f"  {col}: no valid scores")
    print()

    print("-- Feature names --")
    names = processor.get_feature_names()
    for kind, cols in names.items():
        if isinstance(cols, dict):
            print(f"  {kind} ({len(cols)}): {cols}")
        else:
            print(f"  {kind} ({len(cols)}): {cols[:8]}{'...' if len(cols) > 8 else ''}")
    print("=" * 60)
