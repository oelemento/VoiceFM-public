#!/usr/bin/env python3.11
"""Batch convert m4a files to wav for fast loading.

Run this BEFORE finetune_pd.py — converts all m4a files to 16kHz mono wav
so that torchaudio.load() can read them natively (no ffmpeg subprocess needed).
"""
import argparse
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path


def convert_one(m4a_path: Path) -> str:
    wav_path = m4a_path.with_suffix(".wav")
    if wav_path.exists():
        return "skip"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(m4a_path), "-ar", "16000", "-ac", "1",
             "-f", "wav", str(wav_path)],
            capture_output=True, timeout=60,
        )
        if result.returncode != 0:
            return "fail"
        return "ok"
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception:
        return "error"


def main():
    parser = argparse.ArgumentParser(description="Batch convert m4a to wav")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="Root directory containing m4a files")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel ffmpeg processes")
    args = parser.parse_args()

    m4a_files = sorted(args.audio_dir.rglob("*.m4a"))
    print(f"Found {len(m4a_files)} m4a files")

    # Check how many already converted
    already = sum(1 for f in m4a_files if f.with_suffix(".wav").exists())
    print(f"Already converted: {already}, remaining: {len(m4a_files) - already}")

    if already == len(m4a_files):
        print("All files already converted!")
        return

    with Pool(args.workers) as pool:
        results = list(pool.imap_unordered(convert_one, m4a_files, chunksize=100))

    counts = {status: results.count(status) for status in ["ok", "skip", "fail", "timeout", "error"]}
    print(f"Results: {counts}")
    print(f"Total wav files: {already + counts.get('ok', 0)}")

    if counts.get("fail", 0) + counts.get("timeout", 0) + counts.get("error", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
