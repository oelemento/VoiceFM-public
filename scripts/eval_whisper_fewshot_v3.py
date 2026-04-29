#!/usr/bin/env python3
"""Few-shot evaluation of VoiceFM-Whisper on external datasets.

For each dataset and k in [1, 2, 5, 10, 20]:
  - Extract participant-level embeddings (mean-pool recordings)
  - Sample k positive + k negative participants for training
  - Evaluate on remaining participants
  - Repeat 100 times
  - Report mean ± std AUROC

Also extracts frozen Whisper 1280d baseline for comparison.

Usage:
    python -u scripts/eval_whisper_fewshot_v3.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from src.models import build_audio_encoder
from src.utils.preprocessing import load_and_preprocess, MAX_SAMPLES

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEEDS = list(range(42, 47))
KS = [1, 2, 5, 10, 20]
N_TRIALS = 100
BATCH_SIZE = 8
EXT_DIR = PROJECT / "data" / "external"

DATASETS = {
    "Coswara": {
        "meta": EXT_DIR / "coswara" / "metadata.csv",
        "audio": EXT_DIR / "coswara" / "audio",
        "pid_col": "participant_id",
        "label_col": "is_covid",
    },
    "SVD": {
        "meta": EXT_DIR / "svd" / "metadata.csv",
        "audio": EXT_DIR / "svd" / "audio",
        "pid_col": "speaker_id",
        "label_col": "is_pathological",
    },
    "MDVR-KCL": {
        "meta": EXT_DIR / "mdvr_kcl" / "metadata.csv",
        "audio": EXT_DIR / "mdvr_kcl" / "audio",
        "pid_col": "subject_id",
        "label_col": "label",
    },
}


def load_h28_encoder(seed, device):
    ckpt_path = PROJECT / f"checkpoints_exp_whisper_ft4_gsd_v3_seed{seed}" / "best_model.pt"
    if not ckpt_path.exists():
        return None
    with open(PROJECT / "configs" / "model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    model_cfg["audio_encoder"]["type"] = "whisper"
    model_cfg["audio_encoder"]["backbone"] = "openai/whisper-large-v2"
    model_cfg["audio_encoder"]["freeze_backbone"] = True
    model_cfg["audio_encoder"]["unfreeze_last_n"] = 4

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    task_emb_key = "audio_encoder.task_embedding.weight"
    num_task_types = state[task_emb_key].shape[0] if task_emb_key in state else 100
    encoder = build_audio_encoder(config=model_cfg["audio_encoder"], num_task_types=num_task_types)
    ae_state = {k.replace("audio_encoder.", "", 1): v for k, v in state.items()
                if k.startswith("audio_encoder.")}
    encoder.load_state_dict(ae_state)
    return encoder.to(device).eval()


def load_frozen_whisper(device):
    from transformers import WhisperModel, WhisperFeatureExtractor
    model = WhisperModel.from_pretrained("openai/whisper-large-v2", torch_dtype=torch.float32)
    encoder = model.encoder.float().to(device).eval()
    del model.decoder
    for p in encoder.parameters():
        p.requires_grad = False
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")
    return encoder, fe


def extract_h28_embeddings(encoder, meta, audio_dir, pid_col, device):
    """Extract participant-level 256d embeddings."""
    import pandas as pd
    pid_embs = {}
    for _, row in meta.iterrows():
        pid = str(row[pid_col])
        path = audio_dir / row["filename"]
        if not path.exists():
            continue
        try:
            wav = load_and_preprocess(str(path), max_samples=MAX_SAMPLES)
        except Exception:
            continue
        if isinstance(wav, torch.Tensor):
            wav = wav.numpy()
        if len(wav) < 400:
            continue

        padded = np.zeros((1, len(wav)), dtype=np.float32)
        padded[0, :len(wav)] = wav
        masks = np.ones((1, len(wav)), dtype=np.int64)

        with torch.no_grad():
            emb = encoder(
                audio_input_values=torch.tensor(padded, device=device),
                attention_mask=torch.tensor(masks, device=device),
                task_type_ids=torch.zeros(1, dtype=torch.long, device=device),
            )
        pid_embs.setdefault(pid, []).append(emb[0].cpu().numpy())

    return {pid: np.mean(embs, axis=0) for pid, embs in pid_embs.items()}


def extract_frozen_whisper_embeddings(encoder, fe, meta, audio_dir, pid_col, device):
    """Extract participant-level 1280d frozen Whisper embeddings."""
    import pandas as pd
    pid_embs = {}
    for _, row in meta.iterrows():
        pid = str(row[pid_col])
        path = audio_dir / row["filename"]
        if not path.exists():
            continue
        try:
            wav = load_and_preprocess(str(path), max_samples=MAX_SAMPLES)
        except Exception:
            continue
        if isinstance(wav, torch.Tensor):
            wav = wav.numpy()
        if len(wav) < 400:
            continue

        inputs = fe(wav, sampling_rate=16000, return_tensors="pt", padding="max_length")
        mel = inputs.input_features.to(device)
        with torch.no_grad():
            out = encoder(mel, return_dict=True)
            token_len = max(1, min(1500, int(len(wav) / 16000 * 50)))
            pooled = out.last_hidden_state[0, :token_len, :].mean(dim=0).cpu().numpy()
        pid_embs.setdefault(pid, []).append(pooled)

    return {pid: np.mean(embs, axis=0) for pid, embs in pid_embs.items()}


def few_shot_probe(X, y, k, n_trials=100, seed=42):
    """Run n_trials of k-shot participant-level classification."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) < k + 2 or len(neg_idx) < k + 2:
        return {"mean_auroc": float("nan"), "std_auroc": 0, "n_trials": 0, "k": k}

    aurocs = []
    for _ in range(n_trials):
        train_pos = rng.choice(pos_idx, k, replace=False)
        train_neg = rng.choice(neg_idx, k, replace=False)
        train_idx = np.concatenate([train_pos, train_neg])
        test_mask = np.ones(len(y), dtype=bool)
        test_mask[train_idx] = False
        test_idx = np.where(test_mask)[0]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        if len(set(y_test)) < 2:
            continue
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(scaler.fit_transform(X_train), y_train)
        y_prob = clf.predict_proba(scaler.transform(X_test))[:, 1]
        try:
            aurocs.append(roc_auc_score(y_test, y_prob))
        except ValueError:
            continue

    if not aurocs:
        return {"mean_auroc": float("nan"), "std_auroc": 0, "n_trials": 0, "k": k}

    return {
        "mean_auroc": float(np.mean(aurocs)),
        "std_auroc": float(np.std(aurocs)),
        "n_trials": len(aurocs),
        "k": k,
    }


def main():
    import pandas as pd
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    results = {}

    # Load frozen Whisper once
    logger.info("Loading frozen Whisper...")
    fw_encoder, fw_fe = load_frozen_whisper(device)

    for ds_name, ds_cfg in DATASETS.items():
        logger.info("\n=== %s ===", ds_name)
        meta = pd.read_csv(ds_cfg["meta"])
        pid_col = ds_cfg["pid_col"]
        label_col = ds_cfg["label_col"]
        audio_dir = ds_cfg["audio"]
        logger.info("  %d recordings", len(meta))

        # Frozen Whisper embeddings (once)
        logger.info("  Extracting frozen Whisper...")
        fw_embs = extract_frozen_whisper_embeddings(fw_encoder, fw_fe, meta, audio_dir, pid_col, device)
        fw_pids = sorted(fw_embs.keys())
        pid_labels = {str(row[pid_col]): int(row[label_col]) for _, row in meta.iterrows()}
        fw_pids = [p for p in fw_pids if p in pid_labels]
        X_fw = np.array([fw_embs[p] for p in fw_pids])
        y_fw = np.array([pid_labels[p] for p in fw_pids])
        logger.info("  Frozen Whisper: %d participants", len(fw_pids))

        ds_results = {}
        for k in KS:
            fw_res = few_shot_probe(X_fw, y_fw, k)
            ds_results.setdefault(f"k={k}", {})["frozen_whisper"] = fw_res
            logger.info("  k=%d frozen_whisper: %.3f ± %.3f", k, fw_res["mean_auroc"], fw_res["std_auroc"])

        # VoiceFM-Whisper per seed
        for seed in SEEDS:
            logger.info("  VoiceFM-Whisper seed %d...", seed)
            encoder = load_h28_encoder(seed, device)
            if encoder is None:
                continue
            vw_embs = extract_h28_embeddings(encoder, meta, audio_dir, pid_col, device)
            del encoder
            torch.cuda.empty_cache()

            vw_pids = [p for p in sorted(vw_embs.keys()) if p in pid_labels]
            X_vw = np.array([vw_embs[p] for p in vw_pids])
            y_vw = np.array([pid_labels[p] for p in vw_pids])
            logger.info("    %d participants", len(vw_pids))

            for k in KS:
                vw_res = few_shot_probe(X_vw, y_vw, k, seed=seed)
                ds_results[f"k={k}"].setdefault("voicefm_whisper_seeds", []).append(vw_res)
                logger.info("    k=%d: %.3f", k, vw_res["mean_auroc"])

        # Aggregate VoiceFM-Whisper across seeds
        for k in KS:
            seed_aurocs = [s["mean_auroc"] for s in ds_results[f"k={k}"].get("voicefm_whisper_seeds", [])
                           if not np.isnan(s["mean_auroc"])]
            if seed_aurocs:
                ds_results[f"k={k}"]["voicefm_whisper"] = {
                    "mean_auroc": float(np.mean(seed_aurocs)),
                    "std_auroc": float(np.std(seed_aurocs)),
                    "n_seeds": len(seed_aurocs),
                }

        results[ds_name] = ds_results

    del fw_encoder
    torch.cuda.empty_cache()

    out_path = PROJECT / "results_v3" / "fewshot_results_whisper.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nSaved to %s", out_path)

    # Summary
    for ds_name, ds_res in results.items():
        logger.info("\n%s:", ds_name)
        for k in KS:
            vw = ds_res[f"k={k}"].get("voicefm_whisper", {})
            fw = ds_res[f"k={k}"].get("frozen_whisper", {})
            vw_a = vw.get("mean_auroc", float("nan"))
            fw_a = fw.get("mean_auroc", float("nan"))
            logger.info("  k=%2d: VW=%.3f  FW=%.3f  Δ=%+.3f", k, vw_a, fw_a, vw_a - fw_a)


if __name__ == "__main__":
    main()
