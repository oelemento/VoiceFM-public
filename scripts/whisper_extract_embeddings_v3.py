#!/usr/bin/env python3
"""Extract and cache frozen Whisper embeddings for all B2AI recordings.

Saves per-recording 1280d embeddings to results_v3/whisper_recording_embeddings.npz
for reuse by CPU-only analysis scripts.

Usage:
    python -u scripts/whisper_extract_embeddings_v3.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from src.utils.preprocessing import load_and_preprocess, MAX_SAMPLES

BATCH_SIZE = 16

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import WhisperModel, WhisperFeatureExtractor
    logger.info("Loading Whisper large-v2...")
    model = WhisperModel.from_pretrained("openai/whisper-large-v2", torch_dtype=torch.float32)
    encoder = model.encoder.float().to(device).eval()
    del model.decoder
    for p in encoder.parameters():
        p.requires_grad = False
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")

    recordings = pd.read_parquet(PROJECT / "data" / "processed_v3" / "recordings.parquet")
    audio_dir = PROJECT / "data" / "audio"

    all_rec_ids = []
    all_pids = []
    all_names = []
    all_embs = []

    items = [(row["record_id"], row["recording_id"], row.get("recording_name", ""),
              audio_dir / row["audio_filename"])
             for _, row in recordings.iterrows() if (audio_dir / row["audio_filename"]).exists()]

    logger.info("Processing %d recordings...", len(items))

    for batch_start in range(0, len(items), BATCH_SIZE):
        if batch_start % 2000 == 0:
            logger.info("  %d/%d...", batch_start, len(items))

        batch = items[batch_start:batch_start + BATCH_SIZE]
        wavs, pids, rec_ids, names, lengths = [], [], [], [], []

        for pid, rec_id, name, path in batch:
            try:
                wav = load_and_preprocess(str(path), max_samples=MAX_SAMPLES)
            except Exception:
                continue
            if isinstance(wav, torch.Tensor):
                wav = wav.numpy()
            if len(wav) < 400:
                continue
            wavs.append(wav)
            pids.append(pid)
            rec_ids.append(rec_id)
            names.append(name)
            lengths.append(len(wav))

        if not wavs:
            continue

        mel_list = [fe(w, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0) for w in wavs]
        mel = torch.stack(mel_list).to(device)

        with torch.no_grad():
            out = encoder(mel.float(), return_dict=True)
            hidden = out.last_hidden_state

        for i in range(len(wavs)):
            token_len = max(1, int(lengths[i] / MAX_SAMPLES * hidden.shape[1]))
            token_len = min(token_len, hidden.shape[1])
            pooled = hidden[i, :token_len, :].mean(dim=0).cpu().numpy()
            all_embs.append(pooled)
            all_rec_ids.append(rec_ids[i])
            all_pids.append(pids[i])
            all_names.append(names[i])

    # Save
    out_path = PROJECT / "results_v3" / "whisper_recording_embeddings.npz"
    np.savez_compressed(
        out_path,
        embeddings=np.array(all_embs, dtype=np.float32),
        recording_ids=np.array(all_rec_ids, dtype=object),
        participant_ids=np.array(all_pids, dtype=object),
        recording_names=np.array(all_names, dtype=object),
    )
    logger.info("Saved %d embeddings to %s (%.1f MB)",
                len(all_embs), out_path, out_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
