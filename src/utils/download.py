"""Rclone wrapper for downloading audio from an S3-compatible object store.

Configure via environment variables:
- ``VOICEFM_S3_REMOTE``: rclone remote name (default ``s3``); the user must
  configure this remote in their own ``~/.config/rclone/rclone.conf``.
- ``VOICEFM_S3_BUCKET``: bucket holding raw audio files.
- ``VOICEFM_S3_PREFIX``: optional prefix inside the bucket (default empty).

This module does not embed any project- or institution-specific endpoints,
buckets, or credentials. Bridge2AI-Voice users should obtain access through
PhysioNet (see ``data/README.md``) and point these variables at whatever
storage their data-use agreement permits.
"""

import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

WASABI_REMOTE = os.environ.get("VOICEFM_S3_REMOTE", "s3")
RAW_BUCKET = os.environ.get("VOICEFM_S3_BUCKET", "<configure-VOICEFM_S3_BUCKET>")
RAW_PREFIX = os.environ.get("VOICEFM_S3_PREFIX", "")


def download_site(
    site: str,
    dest_dir: str | Path,
    dry_run: bool = False,
    max_files: int | None = None,
) -> None:
    """Download all audio files from a collection site.

    Args:
        site: Collection site name (WCM, MIT, etc.)
        dest_dir: Local destination directory
        dry_run: If True, only list what would be downloaded
        max_files: If set, limit to this many files (for prototyping)
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{RAW_PREFIX}/" if RAW_PREFIX else ""
    source = f"{WASABI_REMOTE}:{RAW_BUCKET}/{prefix}{site}/"
    cmd = ["rclone", "copy", source, str(dest_dir), "--progress", "--transfers", "8"]

    if dry_run:
        cmd = ["rclone", "ls", source]
        if max_files:
            cmd.extend(["--max-count", str(max_files)])
    elif max_files:
        cmd.extend(["--max-transfer", "0", "--cutoff-mode", "soft"])

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=not dry_run)

    if dry_run:
        lines = result.stdout.decode().strip().split("\n")
        if max_files:
            lines = lines[:max_files]
        logger.info(f"Found {len(lines)} files in {site}")
        return lines

    if result.returncode != 0:
        logger.error(f"rclone failed: {result.stderr.decode()}")
        raise RuntimeError(f"Download failed for site {site}")

    logger.info(f"Downloaded {site} to {dest_dir}")


def download_files_by_id(
    recording_ids: list[str],
    site: str,
    dest_dir: str | Path,
) -> dict[str, Path]:
    """Download specific recordings by their IDs.

    Args:
        recording_ids: List of recording UUIDs
        site: Collection site name
        dest_dir: Local destination directory

    Returns:
        Dict mapping recording_id to local file path
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{RAW_PREFIX}/" if RAW_PREFIX else ""
    source = f"{WASABI_REMOTE}:{RAW_BUCKET}/{prefix}{site}/"
    downloaded = {}

    # Create include filter file
    filter_file = dest_dir / ".rclone_filter"
    with open(filter_file, "w") as f:
        for rid in recording_ids:
            f.write(f"+ {rid}.wav\n")
        f.write("- *\n")

    cmd = [
        "rclone", "copy", source, str(dest_dir),
        "--filter-from", str(filter_file),
        "--progress", "--transfers", "8",
    ]

    result = subprocess.run(cmd)
    filter_file.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"Download failed for {len(recording_ids)} files from {site}")

    for rid in recording_ids:
        path = dest_dir / f"{rid}.wav"
        if path.exists():
            downloaded[rid] = path

    logger.info(f"Downloaded {len(downloaded)}/{len(recording_ids)} files from {site}")
    return downloaded


def list_remote_files(site: str) -> list[str]:
    """List all recording filenames in a site bucket.

    Returns:
        List of filenames (e.g., ['UUID1.wav', 'UUID2.wav'])
    """
    prefix = f"{RAW_PREFIX}/" if RAW_PREFIX else ""
    source = f"{WASABI_REMOTE}:{RAW_BUCKET}/{prefix}{site}/"
    result = subprocess.run(
        ["rclone", "lsf", source],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list {site}: {result.stderr}")
    return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    site = sys.argv[1] if len(sys.argv) > 1 else "WCM"
    files = list_remote_files(site)
    print(f"{site}: {len(files)} files")
    for f in files[:5]:
        print(f"  {f}")
