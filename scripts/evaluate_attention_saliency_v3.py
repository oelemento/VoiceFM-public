#!/usr/bin/env python3
"""Visualize attention saliency of VoiceFM vs frozen HuBERT on voice recordings.

Selects exemplar recordings (healthy, breathy, rough, strained based on CAPE-V),
extracts HuBERT attention weights via output_attentions=True (bypassing
AudioEncoder.forward), computes attention rollout, and overlays on mel
spectrograms. Compares VoiceFM (fine-tuned layers 8-11) vs frozen HuBERT.

Usage:
    python3.11 scripts/evaluate_attention_saliency_v3.py \
        --checkpoint checkpoints_exp_d_hard_negatives/best_model.pt \
        --capev-scores data/processed/capev_scores.parquet \
        --baseline --out-dir figures/attention_saliency
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from scipy import stats as scipy_stats
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio_dataset import build_task_type_map
from src.data.sampler import create_participant_splits
from src.models.audio_encoder import AudioEncoder
from src.utils.preprocessing import load_and_preprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


# ── Config / model loading ────────────────────────────────────────────────────

def deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(experiment: str | None = None) -> dict:
    config = {}
    for name in ["model", "data", "train"]:
        path = PROJECT_ROOT / "configs" / f"{name}.yaml"
        if path.exists():
            with open(path) as f:
                config[name] = yaml.safe_load(f)
    if experiment:
        exp_path = PROJECT_ROOT / "configs" / "experiments" / f"{experiment}.yaml"
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {exp_path}")
        with open(exp_path) as f:
            exp_overrides = yaml.safe_load(f)
        for section in ["model", "data", "train"]:
            if section in exp_overrides:
                config[section] = deep_merge(config[section], exp_overrides[section])
                del exp_overrides[section]
        if exp_overrides:
            config["train"] = deep_merge(config["train"], exp_overrides)
    return config


def build_audio_encoder(model_cfg: dict, num_task_types: int, device: torch.device) -> AudioEncoder:
    ae_cfg = model_cfg["audio_encoder"]
    audio_encoder = AudioEncoder(
        backbone=ae_cfg["backbone"],
        freeze_layers=ae_cfg["freeze_layers"],
        projection_dim=ae_cfg["projection_dim"],
        num_task_types=num_task_types,
        spec_augment=False,
        gradient_checkpointing=False,
    )
    return audio_encoder.to(device)


# ── Exemplar selection ────────────────────────────────────────────────────────

def select_exemplars(
    capev_scores: pd.DataFrame,
    participants: pd.DataFrame,
    recordings: pd.DataFrame,
    test_ids: list[str],
    audio_dir: Path,
    n_per_category: int = 5,
) -> pd.DataFrame:
    """Select exemplar prolonged-vowel recordings by CAPE-V extremes.

    Returns DataFrame with: recording_id, record_id, category, and CAPE-V scores.
    """
    # Filter to prolonged vowel in test set
    vowel_mask = recordings["recording_name"].str.lower() == "prolonged vowel"
    vowel = recordings[vowel_mask & recordings["record_id"].isin(test_ids)]

    # Available audio
    available = {p.stem for p in audio_dir.glob("*.wav")}
    vowel = vowel[vowel["recording_id"].isin(available)]

    # One recording per participant (first)
    vowel = vowel.drop_duplicates(subset="record_id", keep="first")

    # Merge with CAPE-V and participants
    vowel = vowel.set_index("record_id")

    exemplars = []

    # Healthy: controls with lowest overall severity
    controls = vowel.index[vowel.index.isin(participants[participants["is_control_participant"] == 1].index)]
    if len(controls) > 0:
        ctrl_with_sev = []
        for pid in controls:
            sev = capev_scores.loc[pid, "overall_severity"] if pid in capev_scores.index else np.nan
            ctrl_with_sev.append((pid, sev if pd.notna(sev) else float("inf")))
        ctrl_with_sev.sort(key=lambda x: x[1])
        for pid, sev in ctrl_with_sev[:n_per_category]:
            exemplars.append({
                "record_id": pid,
                "recording_id": vowel.loc[pid, "recording_id"],
                "category": "healthy",
                "roughness": capev_scores.loc[pid, "roughness"] if pid in capev_scores.index else np.nan,
                "breathiness": capev_scores.loc[pid, "breathiness"] if pid in capev_scores.index else np.nan,
                "strain": capev_scores.loc[pid, "strain"] if pid in capev_scores.index else np.nan,
                "overall_severity": sev,
            })

    # Disease exemplars: top by specific CAPE-V dimension
    disease_pids = vowel.index[vowel.index.isin(
        participants[participants["is_control_participant"] != 1].index
    )]
    disease_with_capev = disease_pids[disease_pids.isin(capev_scores.index)]

    for category, col in [("breathy", "breathiness"), ("rough", "roughness"), ("strained", "strain")]:
        scores = capev_scores.loc[disease_with_capev, col].dropna()
        if len(scores) == 0:
            continue
        top_pids = scores.nlargest(n_per_category).index.tolist()
        for pid in top_pids:
            exemplars.append({
                "record_id": pid,
                "recording_id": vowel.loc[pid, "recording_id"],
                "category": category,
                "roughness": capev_scores.loc[pid, "roughness"] if "roughness" in capev_scores.columns else np.nan,
                "breathiness": capev_scores.loc[pid, "breathiness"] if "breathiness" in capev_scores.columns else np.nan,
                "strain": capev_scores.loc[pid, "strain"] if "strain" in capev_scores.columns else np.nan,
                "overall_severity": capev_scores.loc[pid, "overall_severity"] if "overall_severity" in capev_scores.columns else np.nan,
            })

    return pd.DataFrame(exemplars)


# ── Attention extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_attention(
    audio_encoder: AudioEncoder,
    waveform: torch.Tensor,
    attention_mask: torch.Tensor,
    task_type_id: torch.Tensor,
    device: torch.device,
) -> dict:
    """Extract attention weights by calling hubert directly with output_attentions=True.

    Does NOT modify AudioEncoder.forward(). Accesses audio_encoder.hubert directly.

    Returns:
        dict with keys:
            - layer_attentions: list of (T, T) arrays per layer (head-averaged)
            - rollout: (T,) attention rollout from input to output
            - pooling_weights: (T,) attentive pooling weights
    """
    audio_encoder.eval()
    waveform = waveform.to(device)
    attention_mask = attention_mask.to(device)

    # Step 1: Call HuBERT directly with output_attentions=True
    hubert_output = audio_encoder.hubert(
        input_values=waveform,
        attention_mask=attention_mask,
        output_attentions=True,
        return_dict=True,
    )
    # hubert_output.attentions: tuple of 12 tensors, each (1, 12_heads, T, T)
    attentions = hubert_output.attentions  # tuple of 12 tensors

    # Step 2: Head-average each layer's attention
    layer_attentions = []
    for attn in attentions:
        # (1, heads, T, T) -> (T, T) by averaging over batch and heads
        avg = attn.squeeze(0).mean(dim=0).cpu().numpy()
        layer_attentions.append(avg)

    # Step 3: Attention rollout across all layers
    rollout = compute_attention_rollout(attentions)

    # Step 4: Extract attentive pooling weights via hook
    hidden_states = hubert_output.last_hidden_state  # (1, T, 768)

    # Apply task conditioning (matching AudioEncoder.forward logic)
    if task_type_id is not None:
        task_type_id = task_type_id.to(device)
        if audio_encoder.task_conditioning == "film":
            film_params = audio_encoder.task_embedding(task_type_id)
            scale, shift = film_params.chunk(2, dim=-1)
            hidden_states = hidden_states * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        else:
            task_emb = audio_encoder.task_embedding(task_type_id).unsqueeze(1)
            hidden_states = hidden_states + task_emb

    # Get frame mask
    frame_mask = None
    if attention_mask is not None:
        frame_mask = audio_encoder.hubert._get_feature_vector_attention_mask(
            hidden_states.shape[1], attention_mask,
        )

    # Compute pooling attention scores
    attn_scores = audio_encoder.pooling.attention(hidden_states)  # (1, T, 1)
    if frame_mask is not None:
        mask_expanded = frame_mask.unsqueeze(-1)
        attn_scores = attn_scores.masked_fill(mask_expanded == 0, float("-inf"))
    pooling_weights = F.softmax(attn_scores, dim=1).squeeze(0).squeeze(-1).cpu().numpy()  # (T,)

    return {
        "layer_attentions": layer_attentions,
        "rollout": rollout,
        "pooling_weights": pooling_weights,
        "n_frames": hidden_states.shape[1],
    }


def compute_attention_rollout(attentions: tuple) -> np.ndarray:
    """Compute attention rollout from layer attention matrices.

    Uses residual connection mixing (0.5 * attention + 0.5 * identity).

    Args:
        attentions: tuple of (1, num_heads, T, T) tensors per layer

    Returns:
        rollout: (T,) input attribution from attention rollout
    """
    result = None
    for attn in attentions:
        # Average across heads: (T, T)
        attn_avg = attn.squeeze(0).mean(dim=0)
        T = attn_avg.shape[0]
        I = torch.eye(T, device=attn_avg.device)
        # Mix with residual
        mixed = 0.5 * attn_avg + 0.5 * I
        # Row-normalize
        mixed = mixed / mixed.sum(dim=-1, keepdim=True)
        if result is None:
            result = mixed
        else:
            result = mixed @ result

    # Mean across output positions to get per-input-frame attribution
    rollout = result.mean(dim=0).cpu().numpy()
    return rollout


# ── Mel spectrogram ───────────────────────────────────────────────────────────

def compute_mel_spectrogram(waveform: torch.Tensor, sr: int = 16000) -> np.ndarray:
    """Compute log-mel spectrogram with hop_length=320 (matching HuBERT stride).

    Returns (n_mels, T) numpy array in dB scale.
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=320,
        n_mels=80, f_min=20, f_max=8000,
    )
    mel = mel_transform(waveform)  # (1, n_mels, T)
    mel_db = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel)
    return mel_db.squeeze(0).numpy()  # (n_mels, T)


def compute_rms_energy(waveform: torch.Tensor, hop_length: int = 320) -> np.ndarray:
    """Compute frame-level RMS energy matching HuBERT's stride."""
    wav = waveform.squeeze().numpy()
    n_frames = len(wav) // hop_length
    rms = np.array([
        np.sqrt(np.mean(wav[i * hop_length:(i + 1) * hop_length] ** 2))
        for i in range(n_frames)
    ])
    return rms


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_attention_overlay(
    exemplars: list[dict],
    category: str,
    out_path: Path,
):
    """5×4 grid: mel spectrogram, VoiceFM rollout, HuBERT rollout, pooling weights."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(exemplars)
    col_titles = ["Mel Spectrogram", "VoiceFM Rollout", "HuBERT Rollout", "Pooling Weights"]
    has_hubert = exemplars[0].get("hubert_rollout") is not None
    n_cols = 4 if has_hubert else 3
    if not has_hubert:
        col_titles = ["Mel Spectrogram", "VoiceFM Rollout", "Pooling Weights"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, ex in enumerate(exemplars):
        mel = ex["mel_spectrogram"]
        T_mel = mel.shape[1]

        # Column 0: raw mel spectrogram
        axes[i, 0].imshow(mel, aspect="auto", origin="lower", cmap="gray_r")
        capev_str = f"R={ex.get('roughness', '?'):.0f} B={ex.get('breathiness', '?'):.0f} S={ex.get('strain', '?'):.0f}"
        axes[i, 0].set_ylabel(capev_str, fontsize=8)

        # Column 1: VoiceFM rollout overlay
        rollout = ex["voicefm_rollout"]
        # Interpolate to match mel time axis
        if len(rollout) != T_mel:
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(rollout))
            x_new = np.linspace(0, 1, T_mel)
            rollout = interp1d(x_old, rollout, kind="linear")(x_new)

        axes[i, 1].imshow(mel, aspect="auto", origin="lower", cmap="gray_r")
        rollout_2d = np.broadcast_to(rollout[np.newaxis, :], (mel.shape[0], T_mel))
        axes[i, 1].imshow(rollout_2d, aspect="auto", origin="lower", cmap="Reds", alpha=0.5)

        col_idx = 2
        # Column 2: HuBERT rollout (if available)
        if has_hubert:
            hub_rollout = ex["hubert_rollout"]
            if len(hub_rollout) != T_mel:
                x_old = np.linspace(0, 1, len(hub_rollout))
                x_new = np.linspace(0, 1, T_mel)
                hub_rollout = interp1d(x_old, hub_rollout, kind="linear")(x_new)

            axes[i, col_idx].imshow(mel, aspect="auto", origin="lower", cmap="gray_r")
            hub_2d = np.broadcast_to(hub_rollout[np.newaxis, :], (mel.shape[0], T_mel))
            axes[i, col_idx].imshow(hub_2d, aspect="auto", origin="lower", cmap="Reds", alpha=0.5)
            col_idx += 1

        # Last column: Pooling weights
        pool_w = ex["pooling_weights"]
        if len(pool_w) != T_mel:
            x_old = np.linspace(0, 1, len(pool_w))
            x_new = np.linspace(0, 1, T_mel)
            pool_w = interp1d(x_old, pool_w, kind="linear")(x_new)

        axes[i, col_idx].imshow(mel, aspect="auto", origin="lower", cmap="gray_r")
        pool_2d = np.broadcast_to(pool_w[np.newaxis, :], (mel.shape[0], T_mel))
        axes[i, col_idx].imshow(pool_2d, aspect="auto", origin="lower", cmap="Reds", alpha=0.5)

    # Column titles
    for j, title in enumerate(col_titles[:n_cols]):
        axes[0, j].set_title(title, fontsize=10)

    # Remove tick labels
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(f"Attention Saliency — {category.capitalize()}", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)


def plot_layerwise_attention(
    exemplar_per_category: dict,
    out_path: Path,
):
    """4×12 grid: one exemplar per category, attention at each HuBERT layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = list(exemplar_per_category.keys())
    n_cats = len(categories)
    n_layers = 12

    fig, axes = plt.subplots(n_cats, n_layers, figsize=(2 * n_layers, 2.5 * n_cats))
    if n_cats == 1:
        axes = axes[np.newaxis, :]

    for i, cat in enumerate(categories):
        ex = exemplar_per_category[cat]
        mel = ex["mel_spectrogram"]
        T_mel = mel.shape[1]

        for j, layer_attn in enumerate(ex["layer_attentions"]):
            # layer_attn is (T, T) — take mean across output positions
            frame_attn = layer_attn.mean(axis=0)  # (T,)

            # Interpolate to mel time
            if len(frame_attn) != T_mel:
                x_old = np.linspace(0, 1, len(frame_attn))
                x_new = np.linspace(0, 1, T_mel)
                frame_attn = interp1d(x_old, frame_attn, kind="linear")(x_new)

            axes[i, j].imshow(mel, aspect="auto", origin="lower", cmap="gray_r")
            attn_2d = np.broadcast_to(frame_attn[np.newaxis, :], (mel.shape[0], T_mel))
            axes[i, j].imshow(attn_2d, aspect="auto", origin="lower", cmap="Reds", alpha=0.5)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

            if i == 0:
                axes[i, j].set_title(f"L{j+1}", fontsize=8)

        axes[i, 0].set_ylabel(cat.capitalize(), fontsize=10)

    plt.suptitle("Layer-wise Attention (VoiceFM)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)


def plot_entropy_comparison(
    entropy_data: dict,
    out_path: Path,
):
    """Boxplot: attention entropy (VoiceFM vs HuBERT) by category."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = sorted(entropy_data.keys())
    has_hubert = "hubert_entropy" in entropy_data[categories[0]][0]

    fig, ax = plt.subplots(figsize=(10, 5))

    positions = []
    labels = []
    data_voicefm = []
    data_hubert = []

    for i, cat in enumerate(categories):
        voicefm_ents = [d["voicefm_entropy"] for d in entropy_data[cat]]
        data_voicefm.append(voicefm_ents)
        if has_hubert:
            hubert_ents = [d["hubert_entropy"] for d in entropy_data[cat]]
            data_hubert.append(hubert_ents)

    x = np.arange(len(categories))
    width = 0.35

    bp1 = ax.boxplot(
        data_voicefm, positions=x - width / 2, widths=width * 0.8,
        patch_artist=True, boxprops=dict(facecolor="#2196F3", alpha=0.7),
    )
    if has_hubert:
        bp2 = ax.boxplot(
            data_hubert, positions=x + width / 2, widths=width * 0.8,
            patch_artist=True, boxprops=dict(facecolor="#FF9800", alpha=0.7),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories], fontsize=11)
    ax.set_ylabel("Attention Entropy (nats)", fontsize=11)

    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor="#2196F3", alpha=0.7, label="VoiceFM")]
    if has_hubert:
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor="#FF9800", alpha=0.7, label="HuBERT"))
    ax.legend(handles=legend_elements, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)


def plot_energy_correlation(
    energy_corr_data: dict,
    out_path: Path,
):
    """Bar chart: correlation of attention with RMS energy by category."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = sorted(energy_corr_data.keys())
    has_hubert = "hubert_corr" in energy_corr_data[categories[0]][0]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(categories))
    width = 0.35

    voicefm_means = [np.mean([d["voicefm_corr"] for d in energy_corr_data[cat]]) for cat in categories]
    voicefm_stds = [np.std([d["voicefm_corr"] for d in energy_corr_data[cat]]) for cat in categories]

    ax.bar(x - width / 2, voicefm_means, width, yerr=voicefm_stds,
           label="VoiceFM", color="#2196F3", alpha=0.85, capsize=3)

    if has_hubert:
        hubert_means = [np.mean([d["hubert_corr"] for d in energy_corr_data[cat]]) for cat in categories]
        hubert_stds = [np.std([d["hubert_corr"] for d in energy_corr_data[cat]]) for cat in categories]
        ax.bar(x + width / 2, hubert_means, width, yerr=hubert_stds,
               label="HuBERT", color="#FF9800", alpha=0.85, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories], fontsize=11)
    ax.set_ylabel("Pearson r (attention × RMS energy)", fontsize=11)
    ax.legend(fontsize=10)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Attention saliency analysis for VoiceFM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--capev-scores", type=str, required=True)
    parser.add_argument("--n-exemplars", type=int, default=5)
    parser.add_argument("--baseline", action="store_true", help="Also analyze frozen HuBERT")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="figures/attention_saliency")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    capev_path = Path(args.capev_scores)
    if not capev_path.is_absolute():
        capev_path = PROJECT_ROOT / args.capev_scores
    if not capev_path.exists():
        logger.error("CAPE-V scores not found: %s", args.capev_scores)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.experiment)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # Load data
    data_dir = PROJECT_ROOT / "data" / "processed_v3"
    participants = pd.read_parquet(data_dir / "participants.parquet")
    recordings = pd.read_parquet(data_dir / "recordings.parquet")
    capev_scores = pd.read_parquet(capev_path)
    audio_dir = PROJECT_ROOT / "data" / "audio"

    task_type_map = build_task_type_map(recordings)

    # Get test split
    all_participants = pd.read_parquet(data_dir / "participants.parquet")
    split_cfg = config["data"]["splits"]
    _, _, test_ids = create_participant_splits(
        all_participants,
        train_ratio=split_cfg["train"],
        val_ratio=split_cfg["val"],
        test_ratio=split_cfg["test"],
        seed=split_cfg["seed"],
        stratify_col=split_cfg.get("stratify_by"),
    )
    logger.info("Test set: %d participants", len(test_ids))

    # Select exemplars
    exemplar_df = select_exemplars(
        capev_scores, participants, recordings, test_ids, audio_dir,
        n_per_category=args.n_exemplars,
    )
    exemplar_df.to_csv(out_dir / "exemplar_selection.csv", index=False)
    logger.info("Selected %d exemplars:\n%s", len(exemplar_df), exemplar_df[["record_id", "category"]].to_string())

    # Build VoiceFM audio encoder
    if "whisper" in str(checkpoint_path):
        config["model"]["audio_encoder"]["type"] = "whisper"
        config["model"]["audio_encoder"]["backbone"] = "openai/whisper-large-v2"
        config["model"]["audio_encoder"]["freeze_backbone"] = True
        config["model"]["audio_encoder"]["unfreeze_last_n"] = 4
    audio_encoder = build_audio_encoder(config["model"], len(task_type_map) + 1, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ae_state = {}
    prefix = "audio_encoder."
    for k, v in ckpt["model_state_dict"].items():
        if k.startswith(prefix):
            ae_state[k[len(prefix):]] = v
    audio_encoder.load_state_dict(ae_state)
    # SDPA attention doesn't support output_attentions=True, so reload HuBERT with eager
    # attention and transfer the fine-tuned weights.
    from transformers import HubertModel
    backbone = config["model"]["audio_encoder"]["backbone"]
    eager_hubert = HubertModel.from_pretrained(backbone, attn_implementation="eager").to(device)
    eager_hubert.load_state_dict(audio_encoder.hubert.state_dict())
    audio_encoder.hubert = eager_hubert
    logger.info("Loaded VoiceFM audio encoder from checkpoint (eager attention)")

    # Build frozen HuBERT baseline (if requested)
    hubert_baseline = None
    if args.baseline:
        from transformers import HubertModel
        backbone = config["model"]["audio_encoder"]["backbone"]
        hubert_baseline = HubertModel.from_pretrained(backbone, attn_implementation="eager").to(device)
        hubert_baseline.eval()
        logger.info("Loaded frozen HuBERT baseline: %s", backbone)

    # Process each exemplar
    all_exemplar_data = {cat: [] for cat in ["healthy", "breathy", "rough", "strained"]}
    entropy_data = {cat: [] for cat in ["healthy", "breathy", "rough", "strained"]}
    energy_corr_data = {cat: [] for cat in ["healthy", "breathy", "rough", "strained"]}

    for _, row in exemplar_df.iterrows():
        rec_id = row["recording_id"]
        pid = row["record_id"]
        category = row["category"]
        wav_path = audio_dir / f"{rec_id}.wav"

        logger.info("Processing %s (category=%s, pid=%s)", rec_id, category, pid[:8])

        # Load and preprocess audio (for model input)
        waveform = load_and_preprocess(wav_path)  # 1D tensor (samples,)
        waveform_tensor = waveform.unsqueeze(0)  # (1, samples)
        mask = torch.ones(1, waveform_tensor.shape[1], dtype=torch.long)

        # Task type ID for prolonged vowel
        task_name = "Prolonged vowel"
        task_id = task_type_map.get(task_name, task_type_map.get("Prolonged Vowel", 0))
        task_type_tensor = torch.tensor([task_id], dtype=torch.long)

        # Mel spectrogram (for visualization)
        mel_spec = compute_mel_spectrogram(waveform_tensor)

        # RMS energy
        rms = compute_rms_energy(waveform_tensor)

        # VoiceFM attention
        voicefm_attn = extract_attention(
            audio_encoder, waveform_tensor, mask, task_type_tensor, device,
        )

        # HuBERT baseline attention
        hubert_attn = None
        if hubert_baseline is not None:
            with torch.no_grad():
                hubert_out = hubert_baseline(
                    input_values=waveform_tensor.to(device),
                    attention_mask=mask.to(device),
                    output_attentions=True,
                    return_dict=True,
                )
            hubert_rollout = compute_attention_rollout(hubert_out.attentions)
            hubert_layer_attns = [
                attn.squeeze(0).mean(dim=0).cpu().numpy()
                for attn in hubert_out.attentions
            ]
            hubert_attn = {
                "rollout": hubert_rollout,
                "layer_attentions": hubert_layer_attns,
            }

        # Store exemplar data
        ex_data = {
            "recording_id": rec_id,
            "record_id": pid,
            "category": category,
            "roughness": row.get("roughness", np.nan),
            "breathiness": row.get("breathiness", np.nan),
            "strain": row.get("strain", np.nan),
            "mel_spectrogram": mel_spec,
            "voicefm_rollout": voicefm_attn["rollout"],
            "pooling_weights": voicefm_attn["pooling_weights"],
            "layer_attentions": voicefm_attn["layer_attentions"],
            "hubert_rollout": hubert_attn["rollout"] if hubert_attn else None,
            "hubert_layer_attentions": hubert_attn["layer_attentions"] if hubert_attn else None,
        }
        all_exemplar_data[category].append(ex_data)

        # Quantitative: entropy
        voicefm_rollout = voicefm_attn["rollout"]
        # Normalize to probability distribution
        voicefm_prob = voicefm_rollout / voicefm_rollout.sum()
        voicefm_entropy = float(scipy_stats.entropy(voicefm_prob))

        ent_row = {"voicefm_entropy": voicefm_entropy}
        corr_row = {}

        # Attention-energy correlation
        min_len = min(len(voicefm_rollout), len(rms))
        if min_len > 5:
            r, _ = scipy_stats.pearsonr(voicefm_rollout[:min_len], rms[:min_len])
            corr_row["voicefm_corr"] = float(r)
        else:
            corr_row["voicefm_corr"] = np.nan

        if hubert_attn is not None:
            hub_rollout = hubert_attn["rollout"]
            hub_prob = hub_rollout / hub_rollout.sum()
            ent_row["hubert_entropy"] = float(scipy_stats.entropy(hub_prob))
            min_len = min(len(hub_rollout), len(rms))
            if min_len > 5:
                r, _ = scipy_stats.pearsonr(hub_rollout[:min_len], rms[:min_len])
                corr_row["hubert_corr"] = float(r)
            else:
                corr_row["hubert_corr"] = np.nan

        entropy_data[category].append(ent_row)
        energy_corr_data[category].append(corr_row)

    # ── Generate figures ──────────────────────────────────────────────────
    logger.info("Generating figures...")

    # Attention overlay per category
    for cat, exemplars in all_exemplar_data.items():
        if exemplars:
            plot_attention_overlay(exemplars, cat, out_dir / f"attention_overlay_{cat}.png")

    # Layer-wise attention (one exemplar per category)
    exemplar_per_cat = {}
    for cat, exemplars in all_exemplar_data.items():
        if exemplars:
            exemplar_per_cat[cat] = exemplars[0]
    if exemplar_per_cat:
        plot_layerwise_attention(exemplar_per_cat, out_dir / "layerwise_attention.png")

    # Entropy comparison
    non_empty_entropy = {k: v for k, v in entropy_data.items() if v}
    if non_empty_entropy:
        plot_entropy_comparison(non_empty_entropy, out_dir / "attention_entropy_comparison.png")

    # Energy correlation
    non_empty_corr = {k: v for k, v in energy_corr_data.items() if v}
    if non_empty_corr:
        plot_energy_correlation(non_empty_corr, out_dir / "attention_energy_correlation.png")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  Attention Saliency Analysis Summary")
    print(f"{'=' * 70}")

    for cat in ["healthy", "breathy", "rough", "strained"]:
        if not entropy_data[cat]:
            continue
        vm_ents = [d["voicefm_entropy"] for d in entropy_data[cat]]
        print(f"\n  {cat.capitalize()} (n={len(vm_ents)}):")
        print(f"    VoiceFM entropy: {np.mean(vm_ents):.3f} ± {np.std(vm_ents):.3f}")
        if entropy_data[cat][0].get("hubert_entropy") is not None:
            hub_ents = [d["hubert_entropy"] for d in entropy_data[cat]]
            print(f"    HuBERT  entropy: {np.mean(hub_ents):.3f} ± {np.std(hub_ents):.3f}")
        vm_corrs = [d["voicefm_corr"] for d in energy_corr_data[cat] if not np.isnan(d.get("voicefm_corr", np.nan))]
        if vm_corrs:
            print(f"    VoiceFM-energy r: {np.mean(vm_corrs):.3f} ± {np.std(vm_corrs):.3f}")
        if energy_corr_data[cat] and energy_corr_data[cat][0].get("hubert_corr") is not None:
            hub_corrs = [d["hubert_corr"] for d in energy_corr_data[cat] if not np.isnan(d.get("hubert_corr", np.nan))]
            if hub_corrs:
                print(f"    HuBERT-energy  r: {np.mean(hub_corrs):.3f} ± {np.std(hub_corrs):.3f}")

    print(f"\n{'=' * 70}")

    # Save results
    results = {
        "entropy": {
            cat: [d for d in data]
            for cat, data in entropy_data.items() if data
        },
        "energy_correlation": {
            cat: [d for d in data]
            for cat, data in energy_corr_data.items() if data
        },
        "exemplars": exemplar_df.to_dict(orient="records"),
    }
    results_path = out_dir / "attention_saliency_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
