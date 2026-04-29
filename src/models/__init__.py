from .audio_encoder import AttentivePooling, AudioEncoder
from .clinical_encoder import ClinicalEncoder, FeatureTokenizer
from .voicefm import VoiceFM

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def build_audio_encoder(
    config: dict,
    num_task_types: int,
    spec_augment: bool = False,
    gradient_checkpointing: bool = False,
) -> nn.Module:
    """Build audio encoder from config, selecting HuBERT, HeAR, or Whisper.

    Args:
        config: The ``model.audio_encoder`` config dict.
        num_task_types: Number of recording task types.
        spec_augment: Enable SpecAugment (HuBERT only).
        gradient_checkpointing: Enable gradient checkpointing (HuBERT only).

    Returns:
        Audio encoder module with ``.projection_dim`` attribute and
        ``forward(audio_input_values, attention_mask, task_type_ids)`` interface.
    """
    encoder_type = config.get("type", "hubert")

    if encoder_type == "hear":
        from .hear_encoder import HearAudioEncoder

        encoder = HearAudioEncoder(
            model_name=config.get("backbone", "google/hear-pytorch"),
            projection_dim=config["projection_dim"],
            num_task_types=num_task_types,
            freeze_backbone=config.get("freeze_backbone", True),
            unfreeze_last_n=config.get("unfreeze_last_n", 0),
        )
    elif encoder_type == "whisper":
        from .whisper_encoder import WhisperAudioEncoder

        encoder = WhisperAudioEncoder(
            model_name=config.get("backbone", "openai/whisper-large-v2"),
            projection_dim=config["projection_dim"],
            num_task_types=num_task_types,
            freeze_backbone=config.get("freeze_backbone", True),
            unfreeze_last_n=config.get("unfreeze_last_n", 0),
        )
    else:
        encoder = AudioEncoder(
            backbone=config["backbone"],
            freeze_layers=config["freeze_layers"],
            projection_dim=config["projection_dim"],
            num_task_types=num_task_types,
            spec_augment=spec_augment,
            gradient_checkpointing=gradient_checkpointing,
        )

    # Load pre-trained backbone weights if specified
    pretrained = config.get("pretrained_weights")
    if pretrained and encoder_type == "hubert":
        _load_pretrained_backbone(encoder, pretrained)

    return encoder


def _load_pretrained_backbone(encoder: AudioEncoder, weights_path: str) -> None:
    """Load pre-trained HuBERT + pooling weights into AudioEncoder."""
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    encoder_state = encoder.state_dict()
    matched = 0
    for key, val in state.items():
        if key in encoder_state and encoder_state[key].shape == val.shape:
            encoder_state[key].copy_(val)
            matched += 1
    logger.info(f"Loaded {matched}/{len(state)} pre-trained backbone weights from {weights_path}")


__all__ = [
    "AttentivePooling",
    "AudioEncoder",
    "ClinicalEncoder",
    "FeatureTokenizer",
    "VoiceFM",
    "build_audio_encoder",
]
