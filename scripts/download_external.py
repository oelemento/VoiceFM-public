#!/usr/bin/env python3
"""Download and organize external clinical voice datasets.

Supports:
  - mPower Parkinson's Disease voice study (Synapse syn4993293)
  - Saarbrucken Voice Database (SVD, from Zenodo via sbvoicedb)

Requirements:
  mPower:  pip install synapseclient
  SVD:     pip install sbvoicedb

Usage:
    python scripts/download_external.py --dataset mpower
    python scripts/download_external.py --dataset svd
    python scripts/download_external.py --dataset all
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
TARGET_SR = 16000


def convert_to_wav(src: Path, dst: Path) -> bool:
    """Convert audio file to 16kHz mono WAV using torchaudio."""
    try:
        waveform, sr = torchaudio.load(str(src))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            waveform = resampler(waveform)
        torchaudio.save(str(dst), waveform, TARGET_SR)
        return True
    except Exception as e:
        logger.warning(f"Failed to convert {src}: {e}")
        return False


# ---------------------------------------------------------------------------
# mPower (Synapse)
# ---------------------------------------------------------------------------

VOICE_TABLE_ID = "syn5511444"    # Voice activity recordings
DEMO_TABLE_ID = "syn5511429"     # Demographics survey
BATCH_SIZE = 5000                # Synapse query pagination


def download_mpower():
    """Download mPower voice recordings from Synapse.

    Requires:
      - pip install synapseclient
      - Synapse certified-user account with access to syn4993293
      - SYNAPSE_AUTH_TOKEN env var or cached login (synapse login)

    Downloads voice activity M4A files, converts to 16kHz WAV, and joins
    with demographics table to get diagnosis, age, and gender.
    """
    try:
        import synapseclient
    except ImportError:
        logger.error("Install synapseclient: pip install synapseclient")
        sys.exit(1)

    mpower_dir = EXTERNAL_DIR / "mpower"
    audio_dir = mpower_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    syn = synapseclient.Synapse()
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if token:
        syn.login(authToken=token)
    else:
        syn.login()  # uses cached credentials

    # ---- 1. Download demographics for diagnosis/age/gender ----
    logger.info(f"Querying demographics table ({DEMO_TABLE_ID})...")
    demo_results = syn.tableQuery(f"SELECT * FROM {DEMO_TABLE_ID}")
    demo_df = demo_results.asDataFrame()
    logger.info(f"Demographics: {len(demo_df)} rows")

    # Build healthCode -> demographics lookup
    # professional-diagnosis: True/False for self-reported PD diagnosis
    demo_lookup = {}
    for _, row in demo_df.iterrows():
        hc = row.get("healthCode")
        if pd.isna(hc):
            continue
        demo_lookup[hc] = {
            "is_pd": 1 if row.get("professional-diagnosis", False) else 0,
            "age": int(row["age"]) if not pd.isna(row.get("age", float("nan"))) else -1,
            "gender": row.get("gender", "Unknown"),
        }
    logger.info(f"Demographics lookup: {len(demo_lookup)} unique healthCodes")

    # ---- 2. Download voice recordings in batches ----
    logger.info(f"Querying voice activity table ({VOICE_TABLE_ID})...")

    # Get total row count first
    count_result = syn.tableQuery(
        f'SELECT COUNT(*) FROM {VOICE_TABLE_ID} WHERE "audio_audio.m4a" IS NOT NULL'
    )
    total_rows = count_result.asDataFrame().iloc[0, 0]
    logger.info(f"Total voice recordings with audio: {total_rows}")

    all_voice_rows = []
    all_file_maps = {}

    for offset in range(0, total_rows, BATCH_SIZE):
        logger.info(f"Downloading batch offset={offset}...")
        query = syn.tableQuery(
            f'SELECT recordId, healthCode, medTimepoint, "audio_audio.m4a" '
            f'FROM {VOICE_TABLE_ID} '
            f'WHERE "audio_audio.m4a" IS NOT NULL '
            f'LIMIT {BATCH_SIZE} OFFSET {offset}'
        )
        batch_df = query.asDataFrame()
        all_voice_rows.append(batch_df)

        # Download M4A audio files — returns {fileHandleId: local_path}
        file_map = syn.downloadTableColumns(query, ["audio_audio.m4a"])
        all_file_maps.update(file_map)
        logger.info(f"  Downloaded {len(file_map)} audio files (total: {len(all_file_maps)})")

    voice_df = pd.concat(all_voice_rows, ignore_index=True)
    logger.info(f"Total voice rows: {len(voice_df)}")

    # ---- 3. Convert M4A to 16kHz WAV and build metadata ----
    # Map fileHandleId -> recordId for naming
    handle_to_record = {}
    for _, row in voice_df.iterrows():
        fh = str(row.get("audio_audio.m4a", ""))
        if fh and fh != "nan":
            handle_to_record[fh] = row["recordId"]

    metadata = []
    converted = 0
    for file_handle_id, m4a_path in all_file_maps.items():
        record_id = handle_to_record.get(str(file_handle_id))
        if not record_id:
            continue

        dst = audio_dir / f"{record_id}.wav"
        if not dst.exists():
            if not convert_to_wav(Path(m4a_path), dst):
                continue
        converted += 1

        # Look up demographics
        voice_row = voice_df[voice_df["recordId"] == record_id].iloc[0]
        health_code = voice_row.get("healthCode", "")
        demo = demo_lookup.get(health_code, {})

        gender = demo.get("gender", "Unknown")
        sex = 0 if gender == "Male" else (1 if gender == "Female" else 2)

        metadata.append({
            "record_id": record_id,
            "health_code": health_code,
            "filename": f"{record_id}.wav",
            "is_pd": demo.get("is_pd", 0),
            "age": demo.get("age", -1),
            "sex": sex,
            "med_timepoint": voice_row.get("medTimepoint", ""),
        })

    meta_df = pd.DataFrame(metadata)
    meta_path = mpower_dir / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    logger.info(f"Converted {converted} recordings, metadata ({len(meta_df)} rows) -> {meta_path}")


# ---------------------------------------------------------------------------
# Saarbrucken Voice Database (SVD) via sbvoicedb
# ---------------------------------------------------------------------------

def download_svd():
    """Download Saarbrucken Voice Database using sbvoicedb.

    Uses the sbvoicedb package which auto-downloads from Zenodo (CC-BY-4.0).
    No registration required. Downloads ~17.9 GB incrementally.

    We use only vowel /a/ at normal pitch (a_n) — the standard subset for
    voice pathology detection research (~2,000 recordings).

    Requires: pip install sbvoicedb
    """
    try:
        from sbvoicedb import SbVoiceDb, Recording
    except ImportError:
        logger.error("Install sbvoicedb: pip install sbvoicedb")
        sys.exit(1)

    svd_dir = EXTERNAL_DIR / "svd"
    audio_dir = svd_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Initialize database — downloads incrementally from Zenodo
    logger.info("Initializing SVD database (auto-downloads from Zenodo)...")
    db = SbVoiceDb(
        # Only sustained vowel /a/ at normal pitch — standard research subset
        recording_filter=Recording.utterance == "a_n",
    )

    logger.info(f"SVD sessions downloaded: {db.number_of_sessions_downloaded}/{db.number_of_all_sessions}")

    # Count sessions by type
    session_summary = db.recording_session_summary()
    logger.info(f"SVD sessions: {len(session_summary)}")

    # Extract recordings and convert to 16kHz WAV
    metadata = []
    converted = 0
    errors = 0

    import torch

    for rec in db.iter_recordings():
        speaker_id = rec.session.speaker_id
        session_id = rec.session.id
        session_type = rec.session.type  # "n" = normal, "p" = pathological
        age = rec.session.speaker_age
        gender = rec.session.speaker.gender  # "m" or "w"

        filename = f"s{session_id}_a_n.wav"
        dst = audio_dir / filename

        if not dst.exists():
            try:
                audio_np = rec.nspdata  # numpy array at 50kHz
                sr = rec.rate

                # nspdata can be 2D (samples, channels) — flatten to mono
                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=-1) if audio_np.shape[-1] <= 2 else audio_np.mean(axis=0)

                waveform = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)
                peak = waveform.abs().max()
                if peak > 0:
                    waveform = waveform / peak
                if sr != TARGET_SR:
                    resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                    waveform = resampler(waveform)
                torchaudio.save(str(dst), waveform, TARGET_SR)
            except Exception as e:
                logger.warning(f"Failed to process speaker {speaker_id}: {e}")
                errors += 1
                continue

        converted += 1
        is_pathological = 0 if session_type == "n" else 1
        sex = 0 if gender == "m" else (1 if gender == "w" else 2)

        pathology_names = [p.name for p in rec.session.pathologies]
        diagnosis = "; ".join(pathology_names) if pathology_names else ("healthy" if not is_pathological else "unknown")

        metadata.append({
            "speaker_id": speaker_id,
            "filename": filename,
            "is_pathological": is_pathological,
            "age": int(age) if age and not pd.isna(age) else -1,
            "sex": sex,
            "diagnosis": diagnosis,
        })

    meta_df = pd.DataFrame(metadata)
    meta_path = svd_dir / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    logger.info(
        f"SVD complete: {converted} recordings converted, {errors} errors. "
        f"Metadata ({len(meta_df)} rows) -> {meta_path}"
    )


# ---------------------------------------------------------------------------
# Coswara (COVID-19 voice dataset)
# ---------------------------------------------------------------------------

COSWARA_VALID_COVID = {"positive_mild", "positive_moderate", "positive_asymp"}
COSWARA_HEALTHY = {"healthy"}


def process_coswara():
    """Process Coswara dataset (already cloned from GitHub).

    Expects:
      data/external/coswara/raw/Extracted_data/  (extracted WAVs)
      data/external/coswara/raw/combined_data.csv (metadata)

    Extracts sustained vowel /a/ recordings, resamples to 16kHz,
    and builds metadata with binary COVID label.
    """
    raw_dir = EXTERNAL_DIR / "coswara" / "raw"
    extracted_dir = raw_dir / "Extracted_data"
    meta_csv = raw_dir / "combined_data.csv"

    if not extracted_dir.exists():
        logger.error(
            "Coswara not extracted. Run:\n"
            "  git clone https://github.com/iiscleap/Coswara-Data.git data/external/coswara/raw\n"
            "  cd data/external/coswara/raw && python extract_data.py"
        )
        return

    coswara_dir = EXTERNAL_DIR / "coswara"
    audio_dir = coswara_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    df = pd.read_csv(meta_csv)
    logger.info(f"Coswara metadata: {len(df)} rows")

    # Filter to healthy + COVID positive only
    df_filtered = df[df["covid_status"].isin(COSWARA_VALID_COVID | COSWARA_HEALTHY)].copy()
    logger.info(
        f"Filtered to healthy + COVID positive: {len(df_filtered)} participants "
        f"({df_filtered['covid_status'].value_counts().to_dict()})"
    )

    # Quality annotations for vowel-a
    quality_file = raw_dir / "annotations" / "vowel-a_labels.csv"
    quality_map = {}
    if quality_file.exists():
        qdf = pd.read_csv(quality_file)
        for _, row in qdf.iterrows():
            quality_map[row.iloc[0]] = row.iloc[1]  # participant_id -> quality score

    metadata = []
    converted = 0
    errors = 0

    for _, row in df_filtered.iterrows():
        pid = row["id"]
        covid_status = row["covid_status"]
        is_covid = 1 if covid_status in COSWARA_VALID_COVID else 0

        # Find the participant's vowel-a.wav across date folders
        vowel_path = None
        for date_dir in sorted(extracted_dir.iterdir()):
            candidate = date_dir / pid / "vowel-a.wav"
            if candidate.exists():
                vowel_path = candidate
                break

        if vowel_path is None:
            errors += 1
            continue

        # Skip low quality recordings (score 0 = bad)
        quality = quality_map.get(pid, 2)
        if quality == 0:
            continue

        filename = f"{pid}_vowel_a.wav"
        dst = audio_dir / filename

        if not dst.exists():
            if not convert_to_wav(vowel_path, dst):
                errors += 1
                continue

        converted += 1
        gender = row.get("g", "")
        sex = 0 if gender == "male" else (1 if gender == "female" else 2)
        age = row.get("a", -1)
        try:
            age = int(age) if not pd.isna(age) else -1
        except (ValueError, TypeError):
            age = -1

        metadata.append({
            "participant_id": pid,
            "filename": filename,
            "is_covid": is_covid,
            "covid_status": covid_status,
            "age": age,
            "sex": sex,
        })

    meta_df = pd.DataFrame(metadata)
    meta_path = coswara_dir / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    logger.info(
        f"Coswara complete: {converted} recordings, {errors} errors. "
        f"Metadata ({len(meta_df)} rows) -> {meta_path}"
    )


# ---------------------------------------------------------------------------
# VOICED (PhysioNet)
# ---------------------------------------------------------------------------

def process_voiced():
    """Process VOICED dataset (already downloaded from PhysioNet).

    Expects: data/external/voiced/1.0.0/ (WFDB format files)

    Reads .dat binary audio + .hea headers + -info.txt metadata,
    converts to 16kHz WAV, and builds metadata CSV.
    """
    import struct
    import torch

    wfdb_dir = EXTERNAL_DIR / "voiced" / "1.0.0"
    if not wfdb_dir.exists():
        logger.error(
            "VOICED not downloaded. Run:\n"
            "  wget -r -N -c -np -nH --cut-dirs=2 -P data/external/voiced "
            "https://physionet.org/files/voiced/1.0.0/"
        )
        return

    voiced_dir = EXTERNAL_DIR / "voiced"
    audio_dir = voiced_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    converted = 0
    errors = 0

    # Find all header files
    hea_files = sorted(wfdb_dir.glob("voice*.hea"))
    logger.info(f"VOICED: {len(hea_files)} recordings found")

    for hea_path in hea_files:
        voice_id = hea_path.stem  # e.g., "voice001"

        # Parse header for sample rate and num samples
        with open(hea_path) as f:
            first_line = f.readline().strip().split()
            # Format: record_name num_signals sample_rate num_samples
            sr = int(first_line[2])
            num_samples = int(first_line[3])

        # Read binary .dat file (32-bit signed integers)
        dat_path = hea_path.with_suffix(".dat")
        if not dat_path.exists():
            errors += 1
            continue

        try:
            with open(dat_path, "rb") as f:
                raw = f.read()
            samples = struct.unpack(f"<{num_samples}i", raw)
            audio_np = np.array(samples, dtype=np.float32)

            # Normalize to [-1, 1]
            peak = np.abs(audio_np).max()
            if peak > 0:
                audio_np = audio_np / peak

            waveform = torch.from_numpy(audio_np).unsqueeze(0)  # (1, samples)

            # Resample 8kHz -> 16kHz
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                waveform = resampler(waveform)

            filename = f"{voice_id}.wav"
            dst = audio_dir / filename
            torchaudio.save(str(dst), waveform, TARGET_SR)
            converted += 1
        except Exception as e:
            logger.warning(f"Failed to process {voice_id}: {e}")
            errors += 1
            continue

        # Parse -info.txt for metadata
        info_path = wfdb_dir / f"{voice_id}-info.txt"
        info = {}
        if info_path.exists():
            with open(info_path) as f:
                for line in f:
                    if ":" in line or "\t" in line:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            key = parts[0].rstrip(":").strip()
                            val = parts[1].strip()
                            info[key] = val

        diagnosis = info.get("Diagnosis", "unknown").lower()
        is_pathological = 0 if diagnosis == "healthy" else 1
        gender = info.get("Gender", "")
        sex = 0 if gender == "m" else (1 if gender == "f" else 2)
        age_str = info.get("Age", "-1")
        try:
            age = int(age_str)
        except (ValueError, TypeError):
            age = -1

        vhi = info.get("Voice Handicap Index (VHI) Score", "-1")
        try:
            vhi = int(vhi)
        except (ValueError, TypeError):
            vhi = -1

        metadata.append({
            "voice_id": voice_id,
            "filename": f"{voice_id}.wav",
            "is_pathological": is_pathological,
            "diagnosis": diagnosis,
            "age": age,
            "sex": sex,
            "vhi_score": vhi,
        })

    meta_df = pd.DataFrame(metadata)
    meta_path = voiced_dir / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    logger.info(
        f"VOICED complete: {converted} recordings, {errors} errors. "
        f"Metadata ({len(meta_df)} rows) -> {meta_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Download external voice datasets")
    parser.add_argument(
        "--dataset",
        choices=["mpower", "svd", "coswara", "voiced", "all"],
        required=True,
        help="Which dataset to download",
    )
    args = parser.parse_args()

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("mpower", "all"):
        logger.info("=== Downloading mPower ===")
        download_mpower()

    if args.dataset in ("svd", "all"):
        logger.info("=== Downloading/Processing SVD ===")
        download_svd()

    if args.dataset in ("coswara", "all"):
        logger.info("=== Processing Coswara ===")
        process_coswara()

    if args.dataset in ("voiced", "all"):
        logger.info("=== Processing VOICED ===")
        process_voiced()


if __name__ == "__main__":
    main()
