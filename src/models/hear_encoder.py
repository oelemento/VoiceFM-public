"""
HeAR-based audio encoder for VoiceFM.

Uses Google's Health Acoustic Representations (HeAR) ViT-L model as a
feature extractor. HeAR was trained on 313M 2-second health audio clips
via masked autoencoder self-supervision and produces 512-dim embeddings.

Audio longer than 2s is split into non-overlapping 2-second chunks,
each processed independently, and the resulting embeddings are mean-pooled.

Requires: huggingface-cli login (must accept HeAR license terms first).
Model: https://huggingface.co/google/hear-pytorch
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

HEAR_SAMPLE_RATE = 16000
HEAR_CHUNK_SAMPLES = HEAR_SAMPLE_RATE * 2  # 2 seconds = 32000 samples
MIN_CHUNK_SAMPLES = HEAR_SAMPLE_RATE // 2  # 0.5 seconds minimum
HEAR_POOLER_DIM = 512  # HeAR ViT-L pooler output
MAX_CHUNKS_PER_FORWARD = 64  # sub-batch limit for memory safety


class MelPCENPreprocessor(nn.Module):
    """Convert raw 16 kHz audio to mel-PCEN spectrograms matching HeAR.

    Computes a 128-bin power mel spectrogram with 25 ms frames / 10 ms hops,
    applies Per-Channel Energy Normalization (PCEN), and resizes to the
    (192, 128) input shape expected by HeAR's ViT patch embedding.

    All parameters are frozen (no learnable weights).
    """

    # PCEN parameters from HeAR paper (arXiv:2403.02522)
    PCEN_ALPHA = 0.8
    PCEN_SMOOTH = 0.04
    PCEN_DELTA = 2.0
    PCEN_ROOT = 2.0
    PCEN_FLOOR = 1e-8
    TARGET_SIZE = (192, 128)  # (time, freq) as (height, width)

    def __init__(self) -> None:
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=HEAR_SAMPLE_RATE,
            n_fft=400,       # 25 ms frame
            hop_length=160,  # 10 ms hop
            n_mels=128,
            f_min=0.0,
            f_max=8000.0,
            power=2.0,
        )

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio chunks to mel-PCEN spectrograms.

        Args:
            audio: ``(N, 32000)`` raw 16 kHz mono audio chunks.

        Returns:
            ``(N, 1, 192, 128)`` mel-PCEN spectrograms ready for HeAR.
        """
        mel = self.mel_spec(audio)   # (N, 128, T) where T ≈ 201 for 2s
        mel = self._pcen(mel)        # (N, 128, T)
        mel = mel.transpose(1, 2)    # (N, T, 128) — time on height axis
        mel = mel.unsqueeze(1)       # (N, 1, T, 128)
        mel = F.interpolate(
            mel, size=self.TARGET_SIZE, mode="bilinear", align_corners=False,
        )
        return mel

    def _pcen(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply Per-Channel Energy Normalization.

        PCEN(E, M) = (E / (eps + M)^alpha + delta)^(1/r) - delta^(1/r)
        where M is an IIR-smoothed version of E.
        """
        s = self.PCEN_SMOOTH
        alpha = self.PCEN_ALPHA
        delta = self.PCEN_DELTA
        r = self.PCEN_ROOT
        eps = self.PCEN_FLOOR

        T = mel.shape[2]
        M = torch.empty_like(mel)
        M[:, :, 0] = mel[:, :, 0]
        for t in range(1, T):
            M[:, :, t] = (1 - s) * M[:, :, t - 1] + s * mel[:, :, t]

        return (mel / (eps + M).pow(alpha) + delta).pow(1.0 / r) - delta ** (1.0 / r)


class HearAudioEncoder(nn.Module):
    """HeAR ViT-L backbone with projection head for VoiceFM.

    Processes raw 16 kHz waveforms by:
    1. Chunking into 2-second non-overlapping segments
    2. Converting each chunk to a mel-PCEN spectrogram
    3. Extracting 512-dim embeddings via frozen HeAR ViT-L
    4. Mean-pooling across chunks per sample
    5. Adding optional task-type conditioning
    6. Projecting to the shared contrastive embedding space

    Args:
        model_name: HuggingFace model ID for HeAR.
        projection_dim: Output embedding dimensionality.
        num_task_types: Number of recording task types for conditioning.
        freeze_backbone: Freeze all HeAR parameters (recommended).
        unfreeze_last_n: Unfreeze the last N transformer layers
            (only when ``freeze_backbone=True``).
    """

    def __init__(
        self,
        model_name: str = "google/hear-pytorch",
        projection_dim: int = 256,
        num_task_types: int = 50,
        freeze_backbone: bool = True,
        unfreeze_last_n: int = 0,
    ) -> None:
        super().__init__()

        self.projection_dim = projection_dim
        self.freeze_backbone = freeze_backbone

        # HeAR backbone (ViT-L, ~307M params)
        from transformers import AutoModel

        self.hear = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        if freeze_backbone:
            for param in self.hear.parameters():
                param.requires_grad = False
            if unfreeze_last_n > 0 and hasattr(self.hear, "encoder"):
                for layer in self.hear.encoder.layer[-unfreeze_last_n:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        # Mel-PCEN preprocessing (frozen, no learnable params)
        self.preprocessor = MelPCENPreprocessor()

        # Task-type conditioning (added to pooled embedding before projection)
        self.task_embedding = nn.Embedding(num_task_types, HEAR_POOLER_DIM)

        # Projection head: 512 -> 512 -> 256
        self.projection = nn.Sequential(
            nn.Linear(HEAR_POOLER_DIM, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, projection_dim),
        )

    def forward(
        self,
        audio_input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        task_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode raw waveforms via HeAR into L2-normalized embeddings.

        Args:
            audio_input_values: Raw waveform ``(batch, samples)`` at 16 kHz.
            attention_mask: Binary mask ``(batch, samples)`` for padding.
            task_type_ids: Integer ``(batch,)`` recording task type IDs.

        Returns:
            L2-normalized embeddings ``(batch, projection_dim)``.
        """
        batch_size = audio_input_values.shape[0]
        device = audio_input_values.device

        # 1. Chunk audio into 2-second segments
        chunks, chunk_counts = self._chunk_audio(
            audio_input_values, attention_mask,
        )

        # 2. Preprocess + run HeAR (no grad for frozen backbone)
        # Disable AMP autocast: mel power spectrogram overflows fp16
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", enabled=False)
            if chunks.is_cuda
            else nullcontext()
        )
        with torch.no_grad(), amp_ctx:
            spectrograms = self.preprocessor(chunks.float())
            chunk_embeds = self._run_hear_batched(spectrograms)

        # 3. Mean-pool chunks per sample
        pooled = torch.zeros(batch_size, HEAR_POOLER_DIM, device=device)
        idx = 0
        for i, count in enumerate(chunk_counts):
            pooled[i] = chunk_embeds[idx : idx + count].mean(dim=0)
            idx += count

        # 4. Task-type conditioning
        if task_type_ids is not None:
            pooled = pooled + self.task_embedding(task_type_ids)

        # 5. Project and L2-normalize
        embeddings = self.projection(pooled)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def _run_hear_batched(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Run HeAR in sub-batches to avoid OOM on long recordings."""
        n = spectrograms.shape[0]
        if n <= MAX_CHUNKS_PER_FORWARD:
            out = self.hear(pixel_values=spectrograms, return_dict=True)
            return out.pooler_output

        parts = []
        for start in range(0, n, MAX_CHUNKS_PER_FORWARD):
            sub = spectrograms[start : start + MAX_CHUNKS_PER_FORWARD]
            out = self.hear(pixel_values=sub, return_dict=True)
            parts.append(out.pooler_output)
        return torch.cat(parts, dim=0)

    def _chunk_audio(
        self,
        audio: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[int]]:
        """Split variable-length audio into 2-second chunks.

        Returns:
            chunks: ``(total_chunks, 32000)`` padded audio segments.
            chunk_counts: Per-sample chunk counts (length = batch_size).
        """
        batch_size = audio.shape[0]
        all_chunks: list[torch.Tensor] = []
        chunk_counts: list[int] = []

        for i in range(batch_size):
            length = (
                int(attention_mask[i].sum())
                if attention_mask is not None
                else audio.shape[1]
            )
            wav = audio[i, :length]
            chunks: list[torch.Tensor] = []

            for start in range(0, max(length, 1), HEAR_CHUNK_SAMPLES):
                remaining = length - start
                if remaining < MIN_CHUNK_SAMPLES and chunks:
                    break  # skip very short tail when we already have chunks
                chunk = torch.zeros(
                    HEAR_CHUNK_SAMPLES, device=audio.device, dtype=audio.dtype,
                )
                copy_len = min(max(remaining, 0), HEAR_CHUNK_SAMPLES)
                if copy_len > 0:
                    chunk[:copy_len] = wav[start : start + copy_len]
                chunks.append(chunk)

            if not chunks:
                # Edge case: completely empty audio
                chunks.append(
                    torch.zeros(
                        HEAR_CHUNK_SAMPLES, device=audio.device, dtype=audio.dtype,
                    )
                )

            all_chunks.extend(chunks)
            chunk_counts.append(len(chunks))

        return torch.stack(all_chunks), chunk_counts
