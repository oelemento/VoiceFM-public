"""
Whisper-based audio encoder for VoiceFM.

Uses OpenAI's Whisper large-v2 encoder as a feature extractor.
Whisper was trained on 680K hours of labeled audio for speech recognition
and produces 1280-dim hidden states from 30-second mel spectrograms.

Audio up to 30s is processed directly (matching VoiceFM's max_audio_length).
The encoder output (1500 time steps × 1280 dim) is mean-pooled to a single
1280-dim vector, then projected to the shared contrastive embedding space.

Model: https://huggingface.co/openai/whisper-large-v2
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

WHISPER_SAMPLE_RATE = 16000
WHISPER_MAX_SAMPLES = WHISPER_SAMPLE_RATE * 30  # 30 seconds
WHISPER_HIDDEN_DIM = 1280  # Whisper large-v2 encoder hidden size


class WhisperAudioEncoder(nn.Module):
    """Whisper large-v2 encoder with projection head for VoiceFM.

    Processes raw 16 kHz waveforms by:
    1. Converting to 80-bin log-mel spectrogram via WhisperFeatureExtractor
    2. Extracting 1280-dim hidden states via Whisper encoder (32 layers)
    3. Mean-pooling across time steps (1500 tokens → 1 vector)
    4. Adding optional task-type conditioning
    5. Projecting to the shared contrastive embedding space

    Args:
        model_name: HuggingFace model ID for Whisper.
        projection_dim: Output embedding dimensionality.
        num_task_types: Number of recording task types for conditioning.
        freeze_backbone: Freeze all Whisper encoder parameters.
        unfreeze_last_n: Unfreeze the last N transformer layers.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v2",
        projection_dim: int = 256,
        num_task_types: int = 50,
        freeze_backbone: bool = True,
        unfreeze_last_n: int = 0,
    ) -> None:
        super().__init__()

        self.projection_dim = projection_dim
        self.freeze_backbone = freeze_backbone

        # Load Whisper encoder only (~637M params)
        from transformers import WhisperModel, WhisperFeatureExtractor

        whisper_model = WhisperModel.from_pretrained(model_name)
        self.encoder = whisper_model.encoder
        del whisper_model.decoder  # free decoder memory

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

        self._any_unfrozen = not freeze_backbone or unfreeze_last_n > 0

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            if unfreeze_last_n > 0:
                for layer in self.encoder.layers[-unfreeze_last_n:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        # Task-type conditioning (added to pooled embedding before projection)
        self.task_embedding = nn.Embedding(num_task_types, WHISPER_HIDDEN_DIM)

        # Projection head: 1280 -> 1280 -> 256
        self.projection = nn.Sequential(
            nn.Linear(WHISPER_HIDDEN_DIM, WHISPER_HIDDEN_DIM),
            nn.GELU(),
            nn.LayerNorm(WHISPER_HIDDEN_DIM),
            nn.Linear(WHISPER_HIDDEN_DIM, projection_dim),
        )

    def forward(
        self,
        audio_input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        task_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract embeddings from raw audio.

        Args:
            audio_input_values: (B, T) raw 16 kHz mono waveforms.
            attention_mask: (B, T) binary mask (1 = valid, 0 = padding).
            task_type_ids: (B,) integer task type IDs.

        Returns:
            (B, projection_dim) normalized embeddings.
        """
        device = audio_input_values.device
        batch_size = audio_input_values.shape[0]

        # Convert raw audio to mel spectrogram features
        # Process each sample individually (variable lengths, each padded to 30s)
        mel_list = []
        for i in range(batch_size):
            wav = audio_input_values[i].cpu().numpy()
            if attention_mask is not None:
                valid_len = int(attention_mask[i].sum().item())
                wav = wav[:valid_len]
            mel_list.append(
                self.feature_extractor(
                    wav, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt",
                ).input_features.squeeze(0)
            )
        mel_features = torch.stack(mel_list).to(device)  # (B, n_mels, 3000)

        # When layers are unfrozen, let the trainer's autocast handle precision
        # and allow gradients to flow. When fully frozen, use no_grad() for efficiency.
        ctx = torch.no_grad() if not self._any_unfrozen else contextlib.nullcontext()

        with ctx:
            encoder_out = self.encoder(
                mel_features.float(),
                return_dict=True,
            )

        hidden_states = encoder_out.last_hidden_state  # (B, 1500, 1280)

        # Mean-pool across time dimension
        # If we have attention_mask, compute approximate time mask
        # Whisper always pads to 30s, so we mask based on actual audio length
        if attention_mask is not None:
            # Compute actual lengths from attention_mask
            actual_lengths = attention_mask.sum(dim=1)  # (B,)
            # Map audio sample lengths to encoder token lengths
            # Whisper encoder: 2 conv layers with stride 2 each → 4x downsample
            # Then input is 3000 frames for 30s → 1500 tokens
            token_lengths = (actual_lengths / audio_input_values.shape[1] * hidden_states.shape[1]).long()
            token_lengths = token_lengths.clamp(min=1, max=hidden_states.shape[1])

            # Create time mask
            time_mask = torch.arange(hidden_states.shape[1], device=device).unsqueeze(0)
            time_mask = (time_mask < token_lengths.unsqueeze(1)).float()  # (B, 1500)
            time_mask = time_mask.unsqueeze(-1)  # (B, 1500, 1)

            pooled = (hidden_states * time_mask).sum(dim=1) / time_mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)  # (B, 1280)

        # Add task-type conditioning
        if task_type_ids is not None:
            task_emb = self.task_embedding(task_type_ids)  # (B, 1280)
            pooled = pooled + task_emb

        # Project and normalize
        projected = self.projection(pooled)  # (B, projection_dim)
        return F.normalize(projected, p=2, dim=-1)
